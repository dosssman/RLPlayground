class PPO(object):
    def __init__(self,
        env,
        policy,
        vf,
        qf,
        af,

        loss_clip=.25,

        tb_writer = None):

        self._envs = envs
        self._policy = policy
        self._vf = vf
        self._qf = qf
        self._af = af
        self._loss_clip = loss_clip

    def train():
        pass
