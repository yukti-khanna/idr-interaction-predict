"""Exact sequence resolution, deterministic IDs and explicit row rejections."""
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from urllib.parse import quote
import hashlib
import pandas as pd

AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
PAIR_COLUMNS = ['pair_id', 'AHD_name', 'RGG_name', 'AHD_sequence', 'RGG_sequence']
FLAGS = ['duplicate_pair_id', 'duplicate_sequence_pair', 'missing_sequence',
         'invalid_residue', 'short_for_patch_features']

def clean_sequence(value):
    return ''.join(str(value or '').split()).upper()

def read_table(path):
    return pd.read_csv(path, sep='\t' if Path(path).suffix.lower() in ('.tsv', '.txt') else ',', dtype=str, keep_default_na=False)

def registry_records(path):
    path = Path(path)
    if path.suffix.lower() in ('.fa', '.faa', '.fasta', '.fas', '.fsa'):
        rows = []; name = None; sequence = []
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip(): continue
            if line.startswith('>'):
                if name is not None: rows.append((name, clean_sequence(''.join(sequence))))
                header = line[1:].strip()
                if not header: raise ValueError(f'{path}:{lineno}: empty FASTA identifier')
                name = header.split()[0]; sequence = []
            else:
                if name is None: raise ValueError(f'{path}:{lineno}: sequence before FASTA header')
                sequence.append(line)
        if name is not None: rows.append((name, clean_sequence(''.join(sequence))))
        return rows
    table = read_table(path)
    if not {'name', 'sequence'}.issubset(table.columns):
        raise ValueError(f'{path}: sequence registry table requires name and sequence columns')
    return [(r['name'].strip(), clean_sequence(r['sequence'])) for r in table.to_dict('records')]

def load_registry(paths, role):
    records = []; rows = []; lookup = defaultdict(list)
    for path in paths:
        for name, sequence in registry_records(path):
            records.append((name, sequence, str(path))); lookup[name].append(sequence)
    sequence_names = defaultdict(set)
    for name, sequence, _ in records: sequence_names[sequence].add(name)
    for name, sequence, path in records:
        rows.append({'role':role, 'name':name, 'source_file':path, 'length':len(sequence),
                     'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),
                     'duplicate_name':len(lookup[name]) > 1,
                     'duplicate_sequence_under_other_name':len(sequence_names[sequence]) > 1,
                     'invalid_residue':bool(set(sequence)-AMINO_ACIDS), 'short_for_patch_features':len(sequence)<5})
    return lookup, rows

def canonical_id(ahd, rgg):
    # Preserve historical RGG__AHD order; escape ambiguous separators in names.
    def escape(name):
        value = quote(name, safe='')
        return value.replace('_', '%5F') if '__' in name else value
    return f'{escape(rgg)}__{escape(ahd)}'

def name_list(path):
    return [s.strip() for s in Path(path).read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]

def read_pair_requests(path):
    """Read a table, or headerless .txt pair IDs in historical RGG__AHD order."""
    path = Path(path)
    if path.suffix.lower() == '.txt':
        lines = name_list(path)
        # Retain support for existing tab-delimited manifest files named .txt.
        if not lines or not {'AHD_name', 'RGG_name'}.issubset(lines[0].split('\t')):
            requests = []
            for pair_id in lines:
                row = {'pair_id': pair_id, 'AHD_name': '', 'RGG_name': ''}
                if pair_id.count('__') != 1:
                    row['_pair_parse_error'] = 'invalid_pair_name:expected_RGG_name__AHD_name_with_one_separator'
                else:
                    rgg, ahd = pair_id.split('__')
                    if not rgg or not ahd or any(c.isspace() for c in pair_id):
                        row['_pair_parse_error'] = 'invalid_pair_name:empty_partner_or_internal_whitespace'
                    else:
                        row.update(RGG_name=rgg, AHD_name=ahd)
                requests.append(row)
            return requests
    table = read_table(path)
    if not {'AHD_name', 'RGG_name'}.issubset(table.columns):
        raise ValueError('Pair table requires AHD_name and RGG_name columns; a plain .txt file may instead contain one RGG_name__AHD_name per line')
    return table.to_dict('records')

def resolve_inputs(args):
    registries = {}; registry_audit = []; input_paths = []
    for role, lower in [('AHD','ahd'),('RGG','rgg')]:
        paths = list(getattr(args, lower+'_reference') or [])
        if getattr(args, lower+'_fasta'): paths.append(getattr(args, lower+'_fasta'))
        registries[role], audit = load_registry(paths, role)
        registry_audit.extend(audit); input_paths.extend(paths)
    if args.pairs:
        requests = read_pair_requests(args.pairs); input_paths.append(args.pairs)
    else:
        if args.ahd_list:
            ahds = name_list(args.ahd_list); rggs = name_list(args.rgg_list)
            input_paths.extend([args.ahd_list,args.rgg_list])
        else:
            ahds = list(registries['AHD']); rggs = list(registries['RGG'])
        requests = [{'AHD_name':a,'RGG_name':r} for a,r in product(ahds,rggs)]
    if not requests: raise ValueError('Input requests no pairs')
    # Repeated manifest names are normal; conflicting sequences for one name are not.
    inline = {role:defaultdict(set) for role in ['AHD','RGG']}
    for row in requests:
        for role in inline:
            seq = clean_sequence(row.get(role+'_sequence',''))
            if seq: inline[role][str(row.get(role+'_name','')).strip()].add(seq)
    rows = []
    for index, request in enumerate(requests, 1):
        row = {'input_row':index}; reasons = []
        if request.get('_pair_parse_error'): reasons.append(request['_pair_parse_error'])
        for role in ['AHD','RGG']:
            name = str(request.get(role+'_name','')).strip()
            row[role+'_name'] = name
            supplied = clean_sequence(request.get(role+'_sequence',''))
            candidates = registries[role].get(name,[])
            inline_candidates = inline[role].get(name,set())
            if not name: reasons.append(f'{role}:missing_name')
            if len(candidates)>1: reasons.append(f'{role}:ambiguous_duplicate_registry_name')
            if len(inline_candidates)>1: reasons.append(f'{role}:conflicting_inline_sequences')
            if supplied and candidates and any(supplied != x for x in candidates):
                reasons.append(f'{role}:inline_registry_conflict')
            seq = supplied or (candidates[0] if len(candidates)==1 else '')
            row[role+'_sequence'] = seq; row[role+'_length'] = len(seq)
            row[role+'_sequence_sha256'] = hashlib.sha256(seq.encode()).hexdigest() if seq else ''
            if not seq: reasons.append(f'{role}:missing_sequence')
            if set(seq)-AMINO_ACIDS: reasons.append(f'{role}:invalid_residue:{"".join(sorted(set(seq)-AMINO_ACIDS))}')
            if seq and len(seq)<5: reasons.append(f'{role}:short_for_patch_features')
        row['pair_id'] = str(request.get('pair_id','')).strip() or canonical_id(row['AHD_name'],row['RGG_name'])
        row['failure_reason'] = ';'.join(reasons)
        rows.append(row)
    df = pd.DataFrame(rows)
    df['duplicate_pair_id'] = df.pair_id.duplicated(keep=False)
    df['duplicate_sequence_pair'] = df.duplicated(['AHD_sequence','RGG_sequence'],keep=False)
    for flag in FLAGS[2:]: df[flag] = df.failure_reason.str.contains(flag,regex=False)
    df.loc[df.duplicate_pair_id,'failure_reason'] += ';duplicate_pair_id'
    df['failure_reason'] = df.failure_reason.str.strip(';')
    df['status'] = df.failure_reason.map(lambda s:'rejected' if s else 'valid')
    # Flag identical sequences under distinct names, including manifest-only inputs.
    for role in ['AHD','RGG']:
        distinct = df.groupby(role+'_sequence')[role+'_name'].nunique()
        df[role+'_duplicate_sequence_under_other_name'] = df[role+'_sequence'].map(distinct).gt(1)
    return df, pd.DataFrame(registry_audit,columns=['role','name','source_file','length','sequence_sha256','duplicate_name','duplicate_sequence_under_other_name','invalid_residue','short_for_patch_features']), list(dict.fromkeys(map(str,input_paths)))
