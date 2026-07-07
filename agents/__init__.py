from agents.rql import RQLAgent
from agents.dql import DQLAgent
from agents.dql_hybrid import DQLHybridAgent

agents = dict(
    rql=RQLAgent,
    dql=DQLAgent,
    dql_hybrid=DQLHybridAgent,
)
