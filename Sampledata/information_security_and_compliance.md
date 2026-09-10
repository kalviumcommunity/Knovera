# Knovera Information Security & Compliance Standards
Document ID: SEC-2026-001
Audience: All Staff, Contractors, and Enterprise Partners

## 1. Data Encryption Standards
- **Data at Rest**: All persistent customer data, documents, and dense vector embeddings stored in the MongoDB Atlas database cluster are encrypted using AES-256 GCM encryption.
- **Data in Transit**: All API traffic between web clients, Next.js frontend, and the FastAPI backend must utilize TLS 1.3 encryption. Unencrypted HTTP traffic is rejected.
- **Key Management**: Encryption keys are automatically rotated every 90 calendar days via AWS Key Management Service (KMS) with cloud audit trails enabled.

## 2. Access Control & Multi-Factor Authentication (MFA)
- Mandatory MFA: Multi-factor authentication is strictly enforced for all employee and administrative dashboard logins.
- Session Expiration: Inactive administrative sessions automatically expire after 15 minutes of idle time.
- Password Complexity: Passwords must contain a minimum of 12 characters, including uppercase, lowercase, numbers, and symbols.

## 3. Audit Logging & Compliance Audits
Knovera maintains SOC2 Type II and ISO/IEC 27001 certifications. All user queries, latency metrics, and guardrail interception actions are logged immutably in system audit tables for a minimum retention window of 365 days.
