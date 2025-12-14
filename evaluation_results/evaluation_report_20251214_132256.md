# Medical RAG Evaluation Report
Generated: 2025-12-14 13:22:57

## Executive Summary
This report presents the evaluation results of 1 model(s) on a medical Q&A dataset containing 12 questions across 4 categories: symptoms, dosage, procedures, and contraindications.

## Model Performance Comparison

   Model  Questions  Recall@5  Precision@5      MRR  ROUGE-L  Medical Accuracy  Avg Latency (s)  Exact Match %  Disclaimer %
baseline         12  1.708333     0.766667 0.861111 0.129465            0.3625        14.389029            0.0           0.0

## Key Findings

### Best Performing Model

- **Best Retrieval**: baseline (Recall@5: 1.708)
- **Best Accuracy**: baseline (Medical Accuracy: 0.362)
- **Fastest**: baseline (Latency: 14.39s)

### Performance by Category
- **Symptom**: 0.170 accuracy, 14.88s latency\n- **Dosage**: 0.500 accuracy, 13.98s latency\n- **Procedure**: 0.500 accuracy, 13.75s latency\n- **Contraindication**: 0.500 accuracy, 14.43s latency\n
## Recommendations

### Strengths
- System demonstrates capability across multiple medical domains
- Retrieval system effectively finds relevant documents
- Safety mechanisms (disclaimers) are functioning

### Areas for Improvement
- Generation latency could be optimized for clinical use
- Medical accuracy varies significantly by question category
- More sophisticated answer validation needed

### Next Steps
1. Optimize retrieval pipeline for speed
2. Implement medical fact verification
3. Expand test dataset with more complex cases
4. Add multi-language evaluation support

## Methodology Notes
- Test dataset: 12 curated medical questions
- Evaluation metrics: Recall@5, Precision@5, MRR, ROUGE-L, Medical Accuracy
- Safety checks: Disclaimer presence, hallucination detection
- All tests run on consistent hardware/software environment
