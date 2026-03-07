from neo4j import GraphDatabase
from chromadb import PersistentClient

uri = 'bolt://localhost:7687'
user = 'neo4j'
password = 'GraphDB555'

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    res = session.run('''
        MATCH (n) 
        RETURN labels(n)[0] as Label, count(n) as Count
        ORDER BY Count DESC
    ''')
    print('--- Neo4j Node Counts ---')
    for record in res:
        print(f"{record['Label']}: {record['Count']}")
        
    print('\n--- Neo4j Relationships ---')
    res = session.run('''
        MATCH ()-[r]->()
        RETURN type(r) as Type, count(r) as Count
        ORDER BY Count DESC
    ''')
    for record in res:
        print(f"{record['Type']}: {record['Count']}")

driver.close()

try:
    client = PersistentClient(path='./db/chroma_kg')
    col = client.get_collection('rera_kg_units')
    print(f'\n--- ChromaDB Vectors ---')
    print(f'Total Embeddings: {col.count()}')
except Exception as e:
    print(f'Chroma error: {e}')
