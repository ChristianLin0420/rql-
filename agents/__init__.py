from agents.rql import RQLAgent
from agents.dql import DQLAgent
from agents.dql_hybrid import DQLHybridAgent
from agents.dql_v10 import DQLv10Agent
from agents.dql_v11 import DQLv11Agent
from agents.dql_v11_1 import DQLv11_1Agent
from agents.dql_v11_2 import DQLv11_2Agent

agents = dict(
    rql=RQLAgent,
    dql=DQLAgent,
    dql_hybrid=DQLHybridAgent,
    dql_v10=DQLv10Agent,
    dql_v11=DQLv11Agent,
    dql_v11_1=DQLv11_1Agent,
    dql_v11_2=DQLv11_2Agent,
)
