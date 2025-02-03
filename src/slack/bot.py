import datetime
import uuid
import re
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

from .utils import get_user_name

class SlackBot:
    def __init__(self, slack_token, signing_secret, kg_index, query_engine):
        self.app = App(token=slack_token, signing_secret=signing_secret)
        self.handler = SlackRequestHandler(self.app)
        self.flask_app = Flask(__name__)
        self.kg_index = kg_index
        self.query_engine = query_engine
        self.previous_node = None
        self._setup_routes()

    def _setup_routes(self):
        @self.flask_app.route("/", methods=["POST"])
        def slack_challenge():
            if request.json and "challenge" in request.json:
                return jsonify({"challenge": request.json["challenge"]})
            return self.handler.handle(request)

        @self.app.message()
        def handle_message(message, say):
            self._process_message(message, say)

        @self.flask_app.route("/slack/events", methods=["POST"])
        def slack_events():
            return self.handler.handle(request)

    def _process_message(self, message, say):
        # Handle mentions
        if message.get('blocks'):
            self._handle_mention(message, say)

        # Handle thread replies
        if message.get('thread_ts'):
            self._handle_thread_reply(message)

        # Store message in knowledge graph
        self._store_message(message)

    def _handle_mention(self, message, say):
        for block in message.get('blocks'):
            if block.get('type') == 'rich_text':
                for rich_text_section in block.get('elements', []):
                    for element in rich_text_section.get('elements', []):
                        if element.get('type') == 'user' and element.get('user_id') == self.app.client.auth_test().get("user_id"):
                            for elem in rich_text_section.get('elements', []):
                                if elem.get('type') == 'text':
                                    query = elem.get('text')
                                    response = self._answer_question(query, message)
                                    say(self._clean_markdown(str(response)))

    def _handle_thread_reply(self, message):
        if message.get('parent_user_id') == self.app.client.auth_test().get("user_id"):
            query = message.get('text')
            replies = self.app.client.conversations_replies(
                channel=message.get('channel'),
                ts=message.get('thread_ts')
            )
            response = self._answer_question(query, message, replies)
            self.app.client.chat_postMessage(
                channel=message.get('channel'),
                text=self._clean_markdown(str(response)),
                thread_ts=message.get('thread_ts')
            )

    def _store_message(self, message):
        user_name, _ = get_user_name(self.app.client, message.get('user'))
        dt_object = datetime.datetime.fromtimestamp(float(message.get('ts')))
        formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        text = message.get('text')

        node = TextNode(
            text=text,
            id_=str(uuid.uuid4()),
            metadata={"who": user_name, "when": formatted_time}
        )

        if self.previous_node:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=self.previous_node.node_id)
        self.previous_node = node

        try:
            self.kg_index.insert_nodes([node])
        except Exception as e:
            print(f"Error storing message: {e}")

    def _answer_question(self, query, message, replies=None):
        who_is_asking = get_user_name(self.app.client, message.get('user'))[0]
        return self.query_engine.custom_query(query)

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting from text."""
        # Remove code blocks (both ``` and single `)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Remove links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bullet points
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text

    def run(self, port=3000):
        self.flask_app.run(port=port)
