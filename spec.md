# System Prompt Improvement Specifications

## Current System Analysis

The project currently uses a knowledge graph-based RAG system with the following key components:

- Knowledge extraction from documents using LLM
- Graph-based storage and retrieval
- Slack bot interface for user interactions

## Recommended Prompt Improvements

### 1. Knowledge Extraction Prompt (`KG_TRIPLET_EXTRACT_TMPL`)

Current limitations:

- Single-pass extraction may miss complex relationships
- No domain-specific guidance
- Limited context window utilization

Recommendations:

```markdown
1. Add Domain Context

- Include Block Science specific terminology and concepts
- Add examples of desired entity types and relationships
- Provide guidance on technical vs business entities

2. Implement Multi-pass Extraction

- First pass: Identify core entities and primary relationships
- Second pass: Discover implicit relationships and cross-references
- Final pass: Validate and enrich relationship descriptions

3. Enhance Entity Classification

- Add confidence scores for extracted relationships
- Include temporal aspects of relationships
- Support hierarchical entity types
```

### 2. Query Engine Prompts

Recommendations:

```markdown
1. Add Context-Aware Response Generation

- Include project-specific context in each query
- Maintain conversation history for better context
- Support different response formats based on query type

2. Implement Verification Steps

- Add self-verification of extracted information
- Include confidence scores in responses
- Support citation of source documents

3. Enhanced Error Handling

- Graceful handling of ambiguous queries
- Clear communication of confidence levels
- Ability to request clarification when needed
```

### 3. Community Summary Prompts

Recommendations:

```markdown
1. Improve Clustering Logic

- Add domain-specific clustering criteria
- Support weighted relationships
- Include temporal aspects in community formation

2. Enhanced Summary Generation

- Hierarchical summary structure
- Key concepts extraction
- Cross-community relationships
```

## Implementation Priority

1. Domain Context Enhancement (High Priority)

   - Immediate impact on extraction quality
   - Relatively straightforward to implement

2. Multi-pass Extraction (Medium Priority)

   - Significant improvement in relationship quality
   - Requires careful handling of LLM context window

3. Enhanced Error Handling (Medium Priority)

   - Improves user experience
   - Builds trust in system responses

4. Verification Steps (Low Priority)
   - Important for accuracy
   - Can be implemented incrementally

## Technical Implementation Notes

1. Prompt Template Updates

   - Use formatted JSON for structured data
   - Include example outputs in prompts
   - Add validation rules

2. Configuration Changes

   - Add new environment variables for domain context
   - Configure confidence thresholds
   - Set up logging for prompt performance

3. Integration Points
   - Update GraphRAGExtractor class
   - Modify query engine response handling
   - Enhance Slack bot interaction patterns
