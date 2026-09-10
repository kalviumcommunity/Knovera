# Enterprise Security Guidelines
All customer data and proprietary embeddings in Knovera are stored in the MongoDB Atlas cluster.
AES-256 GCM encryption is enforced for data at rest and in transit.
Automatic key rotation is executed every 90 days with audit logging enabled.
