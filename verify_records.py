"""Verify committed source bindings and accepted public artifact hashes."""
import hashlib,json,pathlib,re
assert __debug__
count=0
for path in pathlib.Path('results').glob('*.json'):
    record=json.loads(path.read_text(encoding='utf-8'))
    for name,expected in record.get('source_sha256',{}).items():
        assert pathlib.Path(name).is_file(),(path,name)
        actual=hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
        assert actual==expected,(path,name,actual,expected)
    for name,expected in record.get('artifacts_sha256',{}).items():
        assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected,(path,name)
    count+=1
assert count
print(count,'research records: committed source and artifact hashes passed')
