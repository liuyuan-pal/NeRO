import argparse
import subprocess
import os
import sys

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)
    print(f"File exists: {path}")

def ensure_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Ensured directory exists: {path}")
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender', type=str, default='blender')
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--material', type=str, required=True)
    parser.add_argument('--hdr', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--trans', dest='trans', action='store_true', default=False)
    args = parser.parse_args()

    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")

    # Check if input files exist
    check_file_exists(args.mesh)
    check_file_exists(args.material)
    check_file_exists(args.hdr)

    # Ensure output directory exists
    output_dir = os.path.dirname(f'data/relight/{args.name}')
    ensure_directory(output_dir)

    # Ensure blender_backend directory exists
    ensure_directory('blender_backend')

    # Check if relight_backend.py exists
    backend_script = 'blender_backend/relight_backend.py'
    check_file_exists(backend_script)

    cmds = [
        args.blender,
        '--background',
        '--python', backend_script,
        '--',
        '--output', f'data/relight/{args.name}',
        '--mesh', args.mesh,
        '--material', args.material,
        '--env_fn', args.hdr,
    ]
    if args.trans:
        cmds.append('--trans')

    print("Executing command:", ' '.join(cmds))

    try:
        subprocess.run(cmds, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()