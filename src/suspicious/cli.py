import argparse
from suspicious.sus import process_text
from suspicious.render import render

def main():
    parser = argparse.ArgumentParser(
        prog='sus', description='Detects possibly suspicious stuff in your source files')
    parser.add_argument('file', nargs='?', help='The file to analyze')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        text = f.read()
        tokens = process_text(text)
        render(tokens, args.file)

if __name__ == '__main__':
    main()