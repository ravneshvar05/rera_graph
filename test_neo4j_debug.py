import logging
import sys
import traceback
from neo4j import GraphDatabase

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

uri = 'neo4j+s://d00bd26c.databases.neo4j.io'
user = 'd00bd26c'
pwd = 's5QXVdHqSkWT1AUgj72jhYEWeen-SxErQmggqC1KSxY'

print("STARTING TEST")
try:
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as session:
        res = session.run("RETURN 1").single()
        print("SUCCESS:", res)
except Exception as e:
    print("FAILED")
    traceback.print_exc(file=sys.stdout)
