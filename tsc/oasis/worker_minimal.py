import os
import sys

def main():
    print(f"PYTHON VERSION: {sys.version}")
    print(f"GRPC_ENABLE_FORK_SUPPORT: {os.environ.get('GRPC_ENABLE_FORK_SUPPORT')}")
    print(f"KMP_DUPLICATE_LIB_OK: {os.environ.get('KMP_DUPLICATE_LIB_OK')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    print("SUCCESS: Minimal worker started.")

if __name__ == "__main__":
    main()
