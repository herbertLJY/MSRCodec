from textless.data.speech_encoder import SpeechEncoder

class HubertModel(object):
    def __init__(self, device):
        dense_model_name = "mhubert-base-25hz"
        quantizer_model_name = "kmeans"
        vocab_size = 500
        deduplicate = False
        self.device = device
        self.encoder = SpeechEncoder.by_name(
            dense_model_name=dense_model_name,
            quantizer_model_name=quantizer_model_name,
            vocab_size=vocab_size,
            deduplicate=deduplicate,
            need_f0=False
        ).to(device)

    def __call__(self, waveform, rate=16000):
        encoded = self.encoder(waveform)
        units = encoded["units"] #.cpu().numpy() 
        return units
