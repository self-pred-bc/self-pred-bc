from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.hiql import HIQLAgent
from agents.ppo import PPOAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.tra import TRAAgent
from agents.byol import BYOLAgent
from agents.byol_min import BYOLMinAgent
from agents.tdsr import TDSRAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    hiql=HIQLAgent,
    ppo=PPOAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    tra=TRAAgent,
    byol=BYOLAgent,
    byol_min=BYOLMinAgent,
    tdsr=TDSRAgent,
)
