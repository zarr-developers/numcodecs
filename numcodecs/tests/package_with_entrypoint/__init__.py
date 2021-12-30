from numcodecs.abc import Codec


class TestCodec(Codec):

    codec_id = "test"

    def encode(self, buf):
        pass

    def decode(self, buf, out=None):
        pass
