def get_frame_saliency(classifier, inner_config=None, **kwargs):
    from art.attacks.evasion import FrameSaliencyAttack

    from armory.utils import config_loading

    attacker = config_loading.load_attack(inner_config, classifier)
    attack = FrameSaliencyAttack(classifier, attacker, **kwargs)
    return attack
