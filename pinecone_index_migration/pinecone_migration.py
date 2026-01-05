import os
import sys
import time
import traceback

# Import the provided PineconeManager class
# The file is named 'pinecone_manager.py' (sic) in the current directory
try:
    from pinecone_manager import PineconeManager
except ImportError:
    # Ensure current directory is in path if running from elsewhere
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from pinecone_manager import PineconeManager

def migrate(source_key, source_index_name, dest_key, dest_index_name):
    # ==========================================
    # 1. CONNECT TO SOURCE
    # ==========================================
    print(f"\n[1/4] Connecting to Source Index: '{source_index_name}'...")
    
    # Configure environment for PineconeManager to pick up source credentials
    os.environ['PINECONE_API_KEY'] = source_key
    os.environ['PINECONE_INDEX_NAME'] = source_index_name
    
    # Instantiate Manager for Source
    source_mgr = PineconeManager()
    
    # Verify Source Index Exists using Manager's method
    available_indexes = source_mgr.list_indexes()
    if source_index_name not in available_indexes:
        print(f"Error: Source index '{source_index_name}' not found. Available: {available_indexes}")
        sys.exit(1)

    # Get Source Dimensions & Metric
    # PineconeManager doesn't expose a direct method to get config of *existing* index, 
    # so we use the underlying client (self.pc) initialized by the manager.
    try:
        source_desc = source_mgr.pc.describe_index(source_index_name)
        source_dim = source_desc.dimension
        source_metric = source_desc.metric
        print(f"      Source Config -> Dimension: {source_dim}, Metric: {source_metric}")
    except Exception as e:
        print(f"Error getting source index details: {e}")
        sys.exit(1)
        
    # Get the raw index object for reading data later
    source_index_obj = source_mgr.get_or_create_index()


    # ==========================================
    # 2. CONNECT TO DESTINATION
    # ==========================================
    print(f"\n[2/4] Connecting to Destination Index: '{dest_index_name}'...")
    
    # Switch environment to Destination credentials
    os.environ['PINECONE_API_KEY'] = dest_key
    os.environ['PINECONE_INDEX_NAME'] = dest_index_name
    
    # Instantiate Manager for Destination
    dest_mgr = PineconeManager()
    
    # Check if Destination Index Exists
    dest_indexes = dest_mgr.list_indexes()
    if dest_index_name in dest_indexes:
        print(f"      Destination index '{dest_index_name}' exists.")
        # Verify Dimensions
        try:
            dest_desc = dest_mgr.pc.describe_index(dest_index_name)
            if dest_desc.dimension != source_dim:
                print(f"❌ Error: Dimension mismatch!")
                print(f"   Source: {source_dim}")
                print(f"   Dest:   {dest_desc.dimension}")
                sys.exit(1)
            print("      Dimensions match.")
        except Exception as e:
            print(f"Warning: Could not verify destination dimensions: {e}")
    else:
        print(f"      Destination index '{dest_index_name}' does not exist.")
        print(f"      Creating index using PineconeManager defaults (Serverless AWS us-east-1)...")
        # Use Manager's create_index method
        dest_mgr.create_index(dimension=source_dim, metric=source_metric)
        
        # Wait for readiness (Manager check is basic, verify with loop)
        while not dest_mgr.pc.describe_index(dest_index_name).status['ready']:
            print("      Waiting for index provisioning...")
            time.sleep(2)
        print("      Index is ready.")

    # ==========================================
    # 3. PREPARE MIGRATION
    # ==========================================
    print(f"\n[3/4] Preparing Migration...")
    
    # Get stats from source to identify namespaces
    source_stats = source_mgr.describe_index_stats()
    namespaces = list(source_stats.namespaces.keys()) if source_stats and source_stats.namespaces else []
    
    # Handle case where default namespace is used but not listed explicitly in keys sometimes, 
    # or if it's the only one.
    if source_stats.total_vector_count > 0 and not namespaces:
        namespaces = ['']
        
    print(f"      Found namespaces: {namespaces}")
    print(f"      Total vectors to migrate: {source_stats.total_vector_count}")

    # ==========================================
    # 4. MIGRATE DATA
    # ==========================================
    print(f"\n[4/4] Starting Data Copy...")
    
    total_migrated = 0
    
    for ns in namespaces:
        ns_label = ns if ns else "(default)"
        print(f"      Processing Namespace: '{ns_label}'")
        
        try:
            # We use the raw source index object to iterate IDs because PineconeManager 
            # does not have a method to list/fetch all IDs.
            # .list() returns a generator of ID lists (pages)
            for id_page in source_index_obj.list(namespace=ns):
                if not id_page:
                    continue
                
                # Fetch vector data (values + metadata)
                # fetch returns a FetchResponse object
                fetch_resp = source_index_obj.fetch(id_page, namespace=ns)
                vectors_map = fetch_resp.get('vectors', {})
                
                # Prepare batch for PineconeManager.upsert_data
                # upsert_data expects a list of dicts: {"id":..., "values":..., "metadata":...}
                batch_data = []
                for vec_id, vec_data in vectors_map.items():
                    batch_data.append({
                        "id": vec_id,
                        "values": vec_data['values'],
                        "metadata": vec_data.get('metadata', {})
                    })
                
                if batch_data:
                    # Use Destination Manager's upsert_data method
                    # It handles batching internally, but we are feeding it paged chunks anyway
                    dest_mgr.upsert_data(batch_data, namespace=ns, batch_size=len(batch_data))
                    count = len(batch_data)
                    total_migrated += count
                    # print(f"      > Processed {count} vectors...")

        except AttributeError:
             print("      ❌ Error: Source index does not support listing IDs (requires Pinecone Serverless or valid Pod setup).")
             break
        except Exception as e:
             print(f"      ❌ Error migrating namespace '{ns_label}': {e}")
             traceback.print_exc()

    print(f"\n✅ Migration Complete.")
    print(f"   Total vectors migrated: {total_migrated}")

def main():
    # Check Env Vars
    # We strip quotes/spaces just in case the .env parsing left them
    source_key = os.environ.get("PINECONE_SOURCE_API_KEY", "").replace('"', '').replace("'", "").strip()
    dest_key = os.environ.get("PINECONE_DEST_API_KEY", "").replace('"', '').replace("'", "").strip()

    if not source_key or not dest_key:
        print("Error: Missing Environment Variables.")
        print("Please ensure 'PINECONE_SOURCE_API_KEY' and 'PINECONE_DEST_API_KEY' are set in your .env file.")
        sys.exit(1)

    # Check Args & Interactive Fallback
    if len(sys.argv) < 2:
        print("No command line arguments provided. Entering interactive mode...")
        source_idx = input("Enter source index name: ").strip()
        if not source_idx:
            print("Error: Source index name is required.")
            sys.exit(1)
        
        dest_idx = input(f"Enter destination index name (Press Enter to use '{source_idx}'): ").strip()
        if not dest_idx:
            dest_idx = source_idx
    else:
        source_idx = sys.argv[1]
        dest_idx = sys.argv[2] if len(sys.argv) > 2 else source_idx
    
    # Run Migration
    migrate(source_key, source_idx, dest_key, dest_idx)

if __name__ == "__main__":
    main()
