"""
Reorganize README.md to move evaluation section up and consolidate content.
"""

def reorganize_readme():
    readme_path = r"c:\Users\USER\Desktop\IR\IR2026\excercise4\README.md"
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Define section boundaries (0-indexed)
    # Section 1: Lines 0-145 - Failure Analysis + ToC
    failure_analysis = lines[0:150]
    
    # Section 2: Lines 150-1250 - Stage 2 & Stage 3 (Technical Implementation)
    technical_impl = lines[150:1250]
    
    # Section 3: Lines 1250-1445 - "Conclusion" with System Architecture Overview
    conclusion_section = lines[1250:1445]
    
    # Section 4: Lines 1445-2071 - Evaluation Methodology + All Evaluations
    evaluation_section = lines[1445:]
    
    # Extract key parts from conclusion for new System Architecture section
    # We'll manually create this
    system_architecture = """
## System Architecture Overview

This project implements a **comprehensive temporal-aware RAG system** for parliamentary and congressional debate analysis, addressing the fundamental limitation of standard RAG systems: **temporal blindness**.

### Three Query Types, Three Solutions

We identified three distinct temporal query patterns, each requiring a specialized retrieval strategy:

| Query Type | Temporal Signal | Example | Solution |
|------------|----------------|---------|----------|
| **Evolution** | Comparative ("how has X changed?", "between 2023 and 2025") | "How did climate policy change over time?" | Double retrieval from early/late windows + structured LLM prompt |
| **Point-in-Time** | Explicit dates ("in 2024", "Q4 2023") | "What healthcare legislation was discussed in 2024?" | Hard filtering (pre-retrieval temporal constraint enforcement) |
| **Recency** | Implicit freshness ("current", "latest", "recent") | "What is the current position on Israel?" | Soft decay re-ranking (half-life recency prior) |

Each query type is automatically detected and routed to the appropriate retrieval strategy, creating a unified temporal RAG system that handles diverse temporal intents without manual configuration.

### Automatic Query Routing

The system automatically detects query intent and routes to the appropriate retrieval strategy:

1. **Evolution patterns** → Double retrieval from early/late windows
2. **Explicit temporal constraints** (via Duckling) → Hard filtering
3. **Recency keywords** ("current", "latest") → Soft decay re-ranking
4. **Default** → Baseline retrieval (no temporal adjustment)

**Key Design Principles:**
- No manual mode selection required
- Graceful degradation when temporal signals are absent
- Corpus-agnostic (works across British Parliament and US Congress)
- Transparent routing with debug output

---

"""
    
    # New structure
    new_readme = (
        # 1. Failure Analysis + ToC
        failure_analysis +
        ["\n"] +
        # 2. System Architecture (extracted from conclusion)
        [system_architecture] +
        # 3. EVALUATION SECTION (moved up)
        evaluation_section +
        ["\n---\n\n"] +
        # 4. Technical Implementation (Stage 2 & 3)
        ["## Technical Implementation\n\n"] +
        ["The following sections describe the technical implementation details of the temporal RAG system.\n\n"] +
        technical_impl
    )
    
    # Remove the old "Conclusion" section since we've extracted what we need
    # (it was in lines 1250-1445 which we're not including)
    
    # Write the reorganized content
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.writelines(new_readme)
    
    print(f"✓ README reorganized successfully")
    print(f"  - Moved evaluation section up")
    print(f"  - Added System Architecture overview")
    print(f"  - Consolidated technical implementation")
    print(f"  - Removed redundant conclusion section")

if __name__ == "__main__":
    reorganize_readme()
