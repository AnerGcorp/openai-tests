import sqlite3
from langchain.tools import Tool

conn = sqlite3.connect("db.sqlite")

def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return  f"The following error occured: {str(err)}"

run_query_tool = Tool.from_function(
    name="run_sqlite_query", 
    description="Run a sqlite query.",
    func=run_sqlite_query
)