import os
from neo4j import GraphDatabase

def wipe_neo4j_database():
    uri = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "blocksofdocs")

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("Successfully wiped Neo4j database")
    except Exception as e:
        print(f"Error wiping database: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    wipe_neo4j_database()
