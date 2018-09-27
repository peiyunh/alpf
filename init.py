import logging
import mxnet as mx

def get_initializer(args):
    if args.initializer == 'xavier':
        return mx.init.Xavier(rnd_type=args.rnd_type, factor_type=args.factor_type, magnitude=args.magnitude)
    elif args.initializer == 'msra':
        return mx.init.MSRAPrelu(args.rnd_type, args.slope)
    elif args.initializer == 'normal':
        raise NotImplementedError('not implemented')

def reinit_network(net, context, args):
    net.collect_params().initialize(get_initializer(args), ctx=context, force_reinit=True)

def init_network(classes, context, args):
    kwargs = {'ctx':context,'pretrained':args.pretrained,'classes':classes,'thumbnail': args.thumbnail}
    net = mx.gluon.model_zoo.vision.get_model(args.architecture, **kwargs)
    net.hybridize()
    net.initialize(get_initializer(args), ctx=context)
    return net

def init_logger(log_file, log_level):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if log_level=='debug' else logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    lsh = logging.StreamHandler()
    lsh.setFormatter(formatter)
    logger.addHandler(lsh)

    lfh = logging.FileHandler(log_file)
    lfh.setFormatter(formatter)
    logger.addHandler(lfh)
    return logger
