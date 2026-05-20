import json

# This is a hacky way to find the translations in the GLB JSON chunk
with open('public/models/office.glb', 'rb') as f:
    f.seek(12)
    chunk_len = int.from_bytes(f.read(4), 'little')
    chunk_type = f.read(4)
    if chunk_type == b'JSON':
        data = f.read(chunk_len)
        gltf = json.loads(data.decode('utf-8'))
        nodes = gltf.get('nodes', [])
        for node in nodes:
            name = node.get('name', '')
            if 'chair' in name.lower() or 'desk' in name.lower() or 'table' in name.lower():
                print(f"{name}: {node.get('translation', [0,0,0])}")
