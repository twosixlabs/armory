"""
Scenario results export functionality for MongoDB
"""

import re

import pymongo
import pymongo.errors

from armory.logs import log

MONGO_PORT = 27017
MONGO_DATABASE = "armory"
MONGO_COLLECTION = "scenario_results"


def send_to_db(
    output: dict,
    host: str,
    port: int = MONGO_PORT,
    database: str = MONGO_DATABASE,
    collection: str = MONGO_COLLECTION,
):
    client = pymongo.MongoClient(host, port)
    db = client[database]
    col = db[collection]
    # strip user/pass off of mongodb url for logging
    tail_of_host = re.findall(r"@([^@]*$)", host)
    if len(tail_of_host) > 0:
        ip = tail_of_host[0]
    else:
        ip = host
    log.info(f"Sending evaluation results to MongoDB instance {ip}:{port}")
    try:
        col.insert_one(output)
    except pymongo.errors.PyMongoError as e:
        log.error(f"Encountered error {e} sending evaluation results to MongoDB")
