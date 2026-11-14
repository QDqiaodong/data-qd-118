DESCRIPTION='Leakage-free rolling-origin demand forecast evaluation'
import argparse, json, sys
from pathlib import Path
from .common import read, save
from .model import analyze
from .vintage import analyze_vintage

def run(source):
    data,digest=read(source)
    return {'schema_version':1,'source_sha256':digest,'result':analyze(data)}

def run_vintage(source):
    data,digest=read(source)
    return {'schema_version':1,'source_sha256':digest,'result':analyze_vintage(data)}

def main():
    parser=argparse.ArgumentParser(description=DESCRIPTION)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source',type=Path,help='observations input (calendar backtest)')
    inputs.add_argument('--vintage-source',type=Path,help='revisions input (vintage backtest)')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    path,build=((args.source,run) if args.source is not None
                else (args.vintage_source,run_vintage))
    try:
        if path.resolve()==args.output.resolve():raise ValueError('input and output must differ')
        report=build(path);save(args.output,report)
    except (ValueError,TypeError,KeyError,OSError,OverflowError,RecursionError) as error:
        print(str(error),file=sys.stderr);return 2
    print(json.dumps(report,sort_keys=True,ensure_ascii=False,allow_nan=False));return 0

if __name__=='__main__':raise SystemExit(main())
