### **What is "Blocks of Docs"?**

**"Blocks of Docs"** is a sophisticated requirements engineering framework designed to streamline the creation, management, and quality assurance of natural language documents. By modularizing document requirements into reusable components called "Blocks," this system enhances reliability, efficiency, and consistency across document production workflows.

### **Core Components**

1. **Blocks**

   - **Definition:** Modular requirements specifications that act as blueprints for creating documents.
   - **Types:**
     - **Simple Blocks:** Fully specify straightforward documents.
     - **Composition Blocks:** Combine multiple simple or other composition blocks to define complex document requirements.
   - **Structure of a Block:**
     - **Name & Description:** Clearly identifies the block and its purpose.
     - **Detailed Requirements:** Specific criteria that the document must meet.
     - **Kits:**
       - **Query Kit (qKit):** Core questions that gather essential information.
       - **Augmentation Kit (aKit):** Additional context or details.
       - **Context Kit (cKit):** Background information to provide necessary context.
     - **Style & Format Specifications:** Guidelines on the document's appearance and formatting.

2. **Docs**

   - **Definition:** The actual documents that are assessed and created based on the specifications outlined in Blocks.
   - **Assessment:** Each Doc is evaluated against its corresponding Block(s) to ensure it meets all specified requirements.

3. **Blocks Database**
   - **Catalog:** A comprehensive repository of all published Blocks, each stored as a JSON record.
   - **Access:** Users can browse, search, and select Blocks from this catalog to define their document requirements.

### **How "Blocks of Docs" Works**

1. **Selecting Blocks:**

   - Users browse the Blocks Database to find suitable Blocks that match their document needs.
   - **Preference for Simplicity:** Simple Blocks are recommended first to ensure ease of use and clarity. Complex Blocks, which are compositions of simpler ones, are used only when necessary.

2. **Defining Document Requirements:**

   - **Simple Documents:** Fully specified by a single simple Block.
   - **Complex Documents:** Defined by composing multiple Blocks to capture intricate requirements.

3. **Workflow Enhancement:**
   - **Expectation Management:** Clearly defined Blocks help align the expectations of all stakeholders involved in the document creation process.
   - **Standardization:** Ensures that all documents adhere to consistent standards and requirements.
   - **Quality Control:** Specific criteria within Blocks facilitate thorough quality checks before document delivery.
   - **Reusability:** Modular Blocks can be reused across different documents, saving time and ensuring consistency.

### **Roles and Responsibilities**

- **Blocks of Docs Librarian:**
  - **Access & Management:** Manages the catalog of Blocks, each represented as a JSON record with detailed specifications.
  - **User Assistance:** Helps users identify the most suitable Blocks for their use cases, prioritizing simplicity and offering concrete advice.
  - **Support Workflow:** Guides users through selecting Blocks, viewing full requirements, and organizing their document creation process.

### **Example Block: "Summary - Prospective"**

- **Type:** Composition Block
- **Specifications:**
  - **Length:** Maximum of 200 words
  - **Format:** Single paragraph, plain text
  - **Purpose:** Provides a concise overview of proposed initiatives, acting as a "flight safety checklist" for project readiness.
- **Components:**
  - **Query Kit (qKit):**
    - Research question/problem
    - Objectives and goals
    - Approach being considered
    - Expected work and next steps
    - Broader implications
  - **Augmentation Kit (aKit):** Additional context like unique aspects of work/team
  - **Context Kit (cKit):** Background information explaining why readers should care
- **Style Requirements:**
  - Concise, laconic language with no technical jargon or excessive acronyms
  - First sentence presents the bottom line upfront
  - Written in a prospective tense

**Purpose & Benefits:**
This block ensures clarity and conciseness, making it ideal for creating "elevator pitch" style summaries that cover all critical aspects of a proposal while remaining accessible to stakeholders.

### **Block Types Overview**

- **Document Summary Blocks:**

  - _Summary - General:_ General-purpose summaries.
  - _Summary - Prospective:_ Forward-looking project summaries.
  - _Summary - Retrospective:_ Summaries of completed work.

- **Proposal-Related Blocks:**

  - _Research Proposal Narrative:_ Detailed descriptions of research projects.
  - _Funding Opportunity Summary:_ Documentation for funding requests.
  - _Travel Request Block:_ Business travel documentation.

- **Technical Documentation Blocks:**

  - _Approach and Methodology:_ Details on work conduct.
  - _Impact and Implications:_ Analysis of outcomes and effects.
  - _Future Work:_ Planning documentation for future initiatives.
  - _Version Control Segment:_ Documentation for version management.

- **Meta Blocks (Blocks about Blocks):**
  - _Block Definition:_ Defines new Blocks.
  - _BlockRequest:_ Requests the creation of new Blocks.

### User Interaction Workflow

1. **Identify Needs:**

   - **User Input:** Users describe their document needs through an interface (e.g., Slack or web portal).
   - **Initial Assessment:** The Librarian bot analyzes the input to understand the required document's scope.

2. **Select Blocks:**

   - **Recommendations:** The Librarian suggests suitable Blocks, prioritizing simple ones for straightforward needs and complex options for intricate requirements.
   - **Alternatives:** If multiple options apply, the Librarian provides brief descriptions to help users choose.

3. **Review Requirements:**

   - **Detailed Overview:** Users can request a full description of selected Blocks, including their structure and criteria.
   - **Customization:** Blocks can be adjusted to better match specific needs.

4. **Answer Requirements:**

   - **Interactive Prompts:** The Librarian guides users through answering questions based on the selected Blocks.
   - **Organized Responses:** User answers are compiled coherently, ensuring completeness.
   - **Validation:** The Librarian checks responses for consistency and helps refine inputs as needed.

5. **Compile and Format Document:**

   - **Content Compilation:** The Librarian assembles information into a cohesive document.
   - **Template Formatting:** Content is formatted according to predefined styles and criteria for professionalism.
   - **Preview:** Users can review and request adjustments before finalization.

6. **Finalize and Deliver Document:**
   - **Delivery:** The finalized document is sent directly to users via Slack, ensuring prompt and seamless access.

## Mapping System (??)

1. **Classify the Question:**

   - **Identify Key Intent:** Detect whether the user is asking _“what,” “why,” “how,” “when,” “who,”_ etc.
   - **Map to Core Elements:**
     - **What** → _Requirement specifics (qKit)_
     - **Why** → _Context (cKit)_
     - **How** → _Method/Approach (qKit or aKit)_
     - **Who/When** → _Augmentation details (aKit)_
   - **Ambiguity Check:** If unclear, ask a short follow-up question to refine the intent.

2. **Link to Block Structure:**

   - **Assign Each Question to a Block Component:** Query Kit, Augmentation Kit, or Context Kit.
   - **Maintain a ‘Master Mapping Table’** that pairs question keywords/phrases with the relevant Block requirements.

3. **Refine with Precedence Rules:**

   - **Simple > Complex:** Always match to the simplest Block or subcomponent first.
   - **Override Logic:** If multiple matches exist, apply a priority list (e.g., _Context > Query > Augmentation_ or any agreed order).

4. **Validate Continuously:**

   - **Inline Prompts:** At each step, confirm with the user if the mapped element still fits the question.
   - **Iterative Updates:** Adjust mappings if new user details suggest a different sub-block is more suitable.

5. **Generate a Structured Response:**
   - **Summarize Mapped Inputs:** Provide an immediate, human-readable outline of which user question maps where.
   - **Highlight Gaps:** Flag any unanswered or ambiguous points for quick resolution.
