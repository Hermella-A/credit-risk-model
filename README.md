# Credit Risk Model – Bati Bank & eCommerce Partner

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

Basel II requires that banks have robust internal risk measurement systems. In credit scoring, this means the model must be **interpretable** (regulators and auditors need to understand why a customer is classified as high‑risk) and **well‑documented** (every step – data sources, feature engineering, assumptions, model choice – must be recorded). A black‑box model would not comply because it would be impossible to validate or challenge its decisions. Hence, our model must balance performance with transparency, and we must maintain a full audit trail.

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy‑based prediction introduce?

The raw data contains transaction records but no historical loan repayment information (default). Therefore we cannot directly train a supervised model. A **proxy variable** (e.g., a "bad customer" label derived from behavioural patterns like failed transactions, high refund rates, or RFM‑based segmentation) is needed as a substitute for default. The business risks include:
- **Proxy bias**: The proxy may not perfectly represent true default risk, leading to misclassification.
- **Regulatory scrutiny**: Using a proxy requires justification; if the proxy is poorly chosen, the model may be rejected by auditors.
- **Operational risk**: Approving or denying credit based on a flawed proxy could cause financial loss or customer churn.

### 3. What are the key trade‑offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high‑performance model (e.g., Gradient Boosting) in a regulated financial context?

| Aspect | Simple (Logistic Regression + WoE) | High‑performance (Gradient Boosting) |
|--------|-------------------------------------|--------------------------------------|
| **Interpretability** | Very high – coefficients directly show impact of each feature. | Low – ensemble of trees is a black box. |
| **Regulatory acceptance** | Higher (easier to explain and document). | Lower – may require additional validation. |
| **Predictive power** | Lower – may miss non‑linear relationships. | Higher – can capture complex interactions. |
| **Risk of overfitting** | Lower (less flexible). | Higher – requires careful tuning and validation. |
| **Development effort** | Lower – fewer hyperparameters. | Higher – more tuning and monitoring needed. |

In a regulated setting, the bank may choose a simple model for initial rollout (to satisfy regulators) and later augment it with a gradient boosting model as a challenger, using explainability tools (SHAP/LIME) to interpret the complex model. The final decision depends on the bank’s risk appetite and the regulator’s stance.