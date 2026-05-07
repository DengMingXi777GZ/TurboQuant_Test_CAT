#!/usr/bin/env python3
"""
Long-context benchmark datasets for testing TurboQuant

These datasets are designed to stress-test:
1. Long context understanding (10k+ tokens)
2. Multi-document reasoning
3. Concurrent request handling

Sources: Based on real long-context benchmark patterns (LongBench, LEval, etc.)
"""

# Dataset 1: Multi-Document QA - 科技公司财报分析
MULTI_DOC_QA = """You are a financial analyst. Based on the following earnings reports from multiple tech companies, answer the questions below.

Document 1 - Apple Inc. Q4 2024:
Apple Inc. reported quarterly revenue of $94.9 billion, up 6% year-over-year. Services revenue reached $25.0 billion, a new all-time record. iPhone revenue was $46.2 billion. The company announced a $110 billion share repurchase program. Apple's active installed base of devices has surpassed 2.2 billion. The Services segment includes the App Store, Apple Music, iCloud, Apple Pay, and Apple TV+.

Document 2 - Microsoft Corporation Q4 2024:
Microsoft reported revenue of $61.9 billion, up 15% year-over-year. Cloud revenue (Azure) grew 29% to $28.5 billion. Productivity and Business Processes revenue was $19.6 billion. LinkedIn revenue increased 10%. Microsoft 365 has 67 million paid subscribers. The company announced a $60 billion share repurchase program and a 10% dividend increase.

Document 3 - Alphabet Inc. Q4 2024:
Alphabet reported revenue of $84.1 billion, up 12% year-over-year. Google Cloud revenue was $11.4 billion, growing 26%. Advertising revenue was $65.5 billion. YouTube advertising revenue was $9.2 billion. The company has over 100 million subscribers to Google One and other subscription services. Waymo completed over 1 million robotaxi rides in 2024.

Document 4 - Amazon.com Q4 2024:
Amazon reported net sales of $187.6 billion, up 10% year-over-year. AWS revenue was $28.8 billion, up 19%. Amazon Prime has over 300 million members globally. Amazon's fulfillment network processed over 2 billion packages in Q4. The company announced plans to invest $11 billion in European logistics expansion.

Document 5 - Meta Platforms Q4 2024:
Meta reported revenue of $40.6 billion, up 25% year-over-year. Family of Apps revenue was $38.5 billion. Reality Labs revenue was $1.1 billion. Meta's ad revenue increased 24%. The company has 3.3 billion daily active users across its platforms. WhatsApp has over 2 billion users.

Questions:
1. Which company had the highest revenue growth rate?
2. Which company's cloud business grew the fastest?
3. What is the combined revenue of all five companies?
4. Which company has the largest user base?
5. Compare the share repurchase programs announced by Apple and Microsoft.

Provide detailed answers with specific numbers.
"""

# Dataset 2: Legal Document Analysis - 复杂的法律合同分析
LEGAL_ANALYSIS = """Analyze the following contract clauses and answer the questions about obligations, liabilities, and termination conditions.

SECTION 1: DEFINITIONS
"Confidential Information" means any non-public information disclosed by either party, including but not limited to technical data, trade secrets, know-how, research, product plans, products, services, customers, markets, software, developments, inventions, processes, formulas, technology, designs, drawings, engineering, hardware configuration information, marketing, finances, or other business information.

SECTION 2: CONFIDENTIALITY OBLIGATIONS
Each party agrees to: (a) hold the other party's Confidential Information in strict confidence; (b) not disclose the Confidential Information to any third parties without prior written consent; (c) use the Confidential Information solely for the purposes of this Agreement; (d) protect the Confidential Information using at least the same degree of care it uses to protect its own confidential information, but in no event less than reasonable care.

SECTION 3: EXCLUSIONS
The obligations of Section 2 shall not apply to information that: (a) is or becomes publicly available through no fault of the receiving party; (b) was properly in the possession of the receiving party prior to disclosure; (c) is independently developed by the receiving party without use of the disclosing party's Confidential Information; (d) is rightfully obtained by the receiving party from a third party without restriction.

SECTION 4: TERM AND TERMINATION
This Agreement shall remain in effect for a period of three (3) years from the Effective Date. Either party may terminate this Agreement: (a) for convenience upon sixty (60) days prior written notice; (b) immediately upon written notice if the other party breaches any material term and fails to cure such breach within thirty (30) days after receipt of written notice.

SECTION 5: LIMITATION OF LIABILITY
IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS, DATA, OR BUSINESS OPPORTUNITIES, REGARDLESS OF WHETHER SUCH DAMAGES WERE FORESEEABLE OR WHETHER EITHER PARTY WAS ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

SECTION 6: INDEMNIFICATION
Each party shall indemnify, defend, and hold harmless the other party from and against any and all claims, damages, losses, costs, and expenses (including reasonable attorneys' fees) arising out of or relating to: (a) any breach of this Agreement by the indemnifying party; (b) any negligent or wrongful act or omission of the indemnifying party.

SECTION 7: INTELLECTUAL PROPERTY
All intellectual property rights in any work product created by either party under this Agreement shall be owned by the party that created such work product. Neither party shall acquire any license or right to use the other party's pre-existing intellectual property except as expressly provided in this Agreement.

SECTION 8: DISPUTE RESOLUTION
Any dispute arising out of or relating to this Agreement shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association. The arbitration shall be conducted in English language. The prevailing party shall be entitled to recover its reasonable attorneys' fees and costs.

Questions:
1. If Party A discovers that information they thought was confidential was actually publicly available, are they still obligated to protect it?
2. If Party A wants to terminate the agreement immediately due to Party B's breach, what must they prove?
3. Can Party A recover lost profits if Party B breaches the agreement?
4. If Party A creates a new invention using Party B's confidential information, who owns the invention?
5. If a dispute arises and goes to arbitration, who pays for the arbitration costs?
"""

# Dataset 3: Academic Paper Comparison - 学术论文对比分析
ACADEMIC_COMPARISON = """Compare and contrast the following three research papers on large language model efficiency.

Paper 1: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
FlashAttention is an exact attention algorithm that is 2-4x faster than standard attention and uses 5-20x less memory. The key innovation is computing attention in a single pass without storing the entire attention matrix to GPU memory. It achieves this by block-wise computation and softmax recalculation. FlashAttention supports variable-length inputs and can be easily integrated with existing transformer models. The algorithm is IO-aware, meaning it considers reads/writes between GPU SRAM and HBM. Experiments show FlashAttention enables training of Transformers with 2x longer context length on the same hardware.

Paper 2: "Sparse Attention Trade-offs in Transformer LLMs"
This paper investigates various sparse attention patterns for reducing the quadratic complexity of self-attention. The authors compare sliding window attention, random attention, and global+local attention. Results show that sliding window attention (Fixed Sparse) achieves 80% of the quality of full attention while reducing memory by 50%. However, sparse attention introduces quality degradation for tasks requiring long-range dependencies. The paper proposes adaptive computation where attention sparsity increases with layer depth. Training stability is a concern with certain sparse patterns.

Paper 3: "Long Context Transformers: A Survey"
This survey covers approaches to extending transformer context lengths beyond 4K tokens. The authors analyze positional interpolation methods (ALiBi, RoPE), efficient attention variants (Linear, Hidden, Performer), and retrieval-augmented approaches. Key findings: (1) Position interpolation generally outperforms extension methods for moderate context extensions (up to 32K). (2) Efficient attention methods sacrifice some quality for length. (3) Retrieval-augmented generation (RAG) can effectively handle arbitrarily long contexts but introduces additional latency. The survey recommends different approaches based on use case: fixed long context for summarization, RAG for question answering over large document collections.

Questions:
1. Which paper focuses on IO-awareness and block-wise computation?
2. What percentage of full attention quality does sliding window attention achieve?
3. Which method is recommended for summarization tasks in Paper 3?
4. How does FlashAttention handle variable-length inputs?
5. What are the main trade-offs between sparse attention and full attention?
"""

# Dataset 4: Medical Case Analysis - 复杂医学病例分析
MEDICAL_ANALYSIS = """A 68-year-old male patient presents with progressive shortness of breath over the past 3 months. He reports a 15-pound unintentional weight loss, night sweats, and chronic fatigue. Past medical history includes hypertension (controlled), type 2 diabetes mellitus, and a 40-pack-year smoking history. He quit smoking 5 years ago.

Physical Examination:
- Vital Signs: BP 142/88 mmHg, HR 98 bpm, RR 22/min, SpO2 92% on room air
- Cardiovascular: Regular rhythm, no murmurs, elevated JVP at 8 cm
- Respiratory: Diminished breath sounds at right base, dullness to percussion
- Abdomen: Soft, non-tender, hepatomegaly 3cm below costal margin
- Lymphatic: Supraclavicular lymphadenopathy present

Laboratory Results:
- CBC: WBC 12,500/μL (neutrophils 75%), Hemoglobin 11.2 g/dL, Platelets 450,000/μL
- CMP: Albumin 2.8 g/dL, ALT 45 U/L, AST 52 U/L, ALP 285 U/L
- LDH: 485 U/L (elevated)
- BNP: 890 pg/mL (elevated)

Imaging:
- Chest CT: Right pleural effusion, 4.3 cm pleural-based mass, mediastinal lymphadenopathy
- PET-CT: FDG uptake in right pleural mass (SUV 12.4), multiple bone lesions, liver lesions

Biopsy Results:
- Immunohistochemistry: WT1 positive, Calretinin positive, D2-40 positive
- Histology: Malignant cells with papillary and tubulopapillary architecture

Questions:
1. What is the most likely diagnosis based on the clinical presentation and biopsy results?
2. Explain the significance of the elevated LDH and BNP levels.
3. What staging workup would you recommend?
4. What are the treatment options for this patient given the stage?
5. What is the prognostic significance of the weight loss and smoking history?
"""

# Dataset 5: Software Architecture Review - 系统架构分析
SOFTWARE_ARCHITECTURE = """Review the following distributed system architecture and answer questions about scalability, reliability, and performance.

MICROSERVICE ARCHITECTURE OVERVIEW:

The system consists of 15 microservices handling e-commerce operations:

1. API Gateway (Kong)
   - Handles load balancing, rate limiting, authentication
   - Processes 50,000 requests/second peak
   - 99.9% availability target
   - Routes to downstream services via service mesh

2. User Service (Go)
   - Manages user accounts, authentication, authorization
   - Uses PostgreSQL with read replicas
   - Caches user sessions in Redis
   - Handles 10,000 concurrent users

3. Product Catalog Service (Java)
   - Product inventory, pricing, search
   - Elasticsearch for product search
   - 500,000 product SKUs
   - Search latency < 100ms p99

4. Order Service (Python)
   - Order processing, fulfillment
   - Kafka for order event streaming
   - PostgreSQL for transaction storage
   - Processes 1,000 orders/second

5. Payment Service (Java)
   - Payment processing, refunds
   - PCI-DSS compliant
   - Integrates with Stripe, PayPal
   - Idempotency keys for retry safety

6. Inventory Service (Rust)
   - Real-time stock management
   - Optimistic locking for concurrency
   - Redis for hot data, PostgreSQL for persistence

7. Notification Service (Node.js)
   - Email, SMS, push notifications
   - Async processing via message queue
   - Template-based rendering

8. Recommendation Service (Python)
   - Collaborative filtering for product recommendations
   - TensorFlow models served via TensorFlow Serving
   - 50ms latency p99

9. Analytics Service (Scala)
   - Clickstream processing with Apache Flink
   - Real-time dashboards
   - Data lake integration with Snowflake

DATABASE INFRASTRUCTURE:
- PostgreSQL: 3 master clusters with async replication
- Redis: 6-node cluster, 500GB total memory
- Elasticsearch: 9-node cluster for search
- Kafka: 12-broker cluster for event streaming
- Object Storage: 100TB for media files

Questions:
1. What is the single point of failure in this architecture?
2. How would you handle a 10x traffic spike during a flash sale?
3. What changes would you make to achieve 99.99% availability?
4. How do you ensure data consistency across services during distributed transactions?
5. What monitoring and alerting would you implement for the payment service?
"""

# All test datasets with metadata
TEST_DATASETS = {
    "multi_doc_qa": {
        "name": "Multi-Document QA (Financial Reports)",
        "prompt": MULTI_DOC_QA,
        "expected_tokens": 2500,
        "difficulty": "hard",
        "category": "reasoning"
    },
    "legal_analysis": {
        "name": "Legal Contract Analysis",
        "prompt": LEGAL_ANALYSIS,
        "expected_tokens": 2000,
        "difficulty": "hard",
        "category": "comprehension"
    },
    "academic_comparison": {
        "name": "Academic Paper Comparison",
        "prompt": ACADEMIC_COMPARISON,
        "expected_tokens": 1800,
        "difficulty": "medium",
        "category": "reasoning"
    },
    "medical_analysis": {
        "name": "Medical Case Analysis",
        "prompt": MEDICAL_ANALYSIS,
        "expected_tokens": 1500,
        "difficulty": "hard",
        "category": "expert_knowledge"
    },
    "software_architecture": {
        "name": "Software Architecture Review",
        "prompt": SOFTWARE_ARCHITECTURE,
        "expected_tokens": 2200,
        "difficulty": "hard",
        "category": "technical"
    }
}

def get_test_prompt(dataset_key=None):
    """Get a test prompt, optionally filtered by category or difficulty"""
    import random

    if dataset_key and dataset_key in TEST_DATASETS:
        return TEST_DATASETS[dataset_key]["prompt"]

    if dataset_key == "random":
        return random.choice(list(TEST_DATASETS.values()))["prompt"]

    if dataset_key == "hard":
        hard_datasets = [d for d in TEST_DATASETS.values() if d["difficulty"] == "hard"]
        return random.choice(hard_datasets)["prompt"]

    return TEST_DATASETS["multi_doc_qa"]["prompt"]

if __name__ == "__main__":
    print("Available test datasets:")
    for key, data in TEST_DATASETS.items():
        print(f"  {key}: {data['name']} ({data['difficulty']}, ~{data['expected_tokens']} tokens)")
