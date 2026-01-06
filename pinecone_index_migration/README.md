# Pinecone Index Migration Script

## Purpose
This script facilitates the complete migration of vector data from one Pinecone index to another. It is designed to handle the complexity of reading vector data, metadata, and namespaces from a source index and replicating it faithfully in a destination index. This tool is useful for backups, environment promotions (e.g., dev to prod), or consolidating indexes.



## Technical Migration Flow
1.  **Analyze Source**: Connects to the source index to retrieve statistical metadata, including total vector counts and namespace distribution.
2.  **Prepare Destination**: Provisions or validates the destination index to ensure it is ready to receive vectors.
3.  **Iterate & Transfer**:
    *   Iterates through every namespace found in the source.
    *   Lists all vector IDs within that namespace.
    *   Fetches vector values and metadata for those IDs.
    *   Upserts the data into the corresponding namespace in the destination index.

This tool abstracts away the manual effort of scripting API calls for vector migration, providing a reliable, automated solution.
