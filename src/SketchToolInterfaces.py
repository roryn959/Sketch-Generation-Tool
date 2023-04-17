# Defines pseudo-interfaces for use in SketchTool.py

class ModelFactoryInterface:
    ''' Class used by ModelHandler to acces models '''

    def getSession(self):
        ''' Returns the session used by the loaded model '''
        raise NotImplementedError()

    def getEncodeModel(self):
        ''' Returns the model used to encode sketches
            onto latent space. May be same model as
            decoder '''
        raise NotImplementedError()

    def getDecodeModel(self):
        ''' Returns the model used to decode latent vectors
            into sketches. May be same model as encoder '''
        raise NotImplementedError()


class ModelHandlerInterface:
    ''' Class used by GUI to abstract task of interacting
        with models '''

    def getMaxSeqLen(self):
        ''' Returns maximum number of strokes expected in a sketch '''
        raise NotImplementedError()

    def sketchToLatent(self, sketch):
        ''' Encode sketch into latent vector '''
        raise NotImplementedError()

    def generateFromLatent(self, z, existing_strokes):
        ''' Generate a sketch given a latent space. If existing strokes
            given, complete sketch. Otherwise generate from scratch. '''
        raise NotImplementedError()

    def getLatentSize(self):
        ''' Return size of latent vector expected by model '''
        raise NotImplementedError()
