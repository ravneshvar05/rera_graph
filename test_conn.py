from neo4j import GraphDatabase
import traceback
import sys
try:
    uri = 'neo4j+s://d00bd26c.databases.neo4j.io'
    user = 'd00bd26c'
    pwd = 's5QXVdHqSkWT1AUgj72jhYEWeen-SxErQmggqC1KSxY'
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    print("SUCCESS")
except Exception as e:
    print("FAILURE")
    traceback.print_exc(file=sys.stdout)
