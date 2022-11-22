import sqlite3
import random

con = sqlite3.connect("./database/data.db")
cur = con.cursor()

res = cur.execute("select count(*) > 0 from sqlite_master sm where sm.tbl_name = 'sensor'")
exists = res.fetchone()[0]

if not exists:
    cur.execute("CREATE TABLE sensor ( \
    id INT PRIMARY KEY,\
    a_x FLOAT, \
    a_y FLOAT, \
    a_z FLOAT, \
    )")


def insert(x,y,z):
    cur.execute(f"INSERT INTO sensor VALUES ({0},{x},{y},{z})")

x,y,z = random.random(), random.random(),random.random()

insert(x,y,z)
insert(x,y,z)
insert(x,y,z)

res = cur.execute("select * from sensor")

print(res.fetchall())