import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender', type=str)
    parser.add_argument('--mesh', type=str)
    parser.add_argument('--material', type=str)
    parser.add_argument('--hdr', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--trans', dest='trans', action='store_true', default=False)
    args = parser.parse_args()

    cmds = [
        args.blender, '--background', '--python', 'blender_backend/relight_backend.py', '--',
        '--output', f'data/relight/{args.name}',
        '--mesh', args.mesh,
        '--material', args.material,
        '--env_fn', args.hdr,
    ]
    if args.trans:
        cmds.append('--trans')
    subprocess.run(cmds)


if __name__ == "__main__":
    main()
