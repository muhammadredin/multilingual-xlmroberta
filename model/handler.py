import logging
from abc import ABC
from collections.abc import Iterable
import transformers
import torch

import numpy as np
from ts.metrics.dimension import Dimension

logger = logging.getLogger(__name__)

from ts.torch_handler.base_handler import BaseHandler


logger = logging.getLogger(__name__)
logger.info("Transformers version %s",transformers.__version__)

class CustomHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence classification.
    """

    def __init__(self):
        super(CustomHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        self.model = transformers.XLMRobertaForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(model_dir)

        self.model.eval()

        logger.info(
            "Transformer model from path %s loaded successfully", model_dir
        )
        
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            request = data.get("data")
            if request is None:
                request = data.get("body")
            if isinstance(request, (bytes, bytearray)):
                request = request.decode('utf-8')

            input_text = request['text']
            logger.info("Received text: '%s'", input_text)

            # preprocessing text for sequence_classification and token_classification.
            inputs = self.tokenizer.encode_plus(input_text, max_length=128, pad_to_max_length=True, add_special_tokens=True, return_attention_mask = True, return_tensors='pt')
            
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat((attention_mask_batch, attention_mask), 0)
        
        input_ids_batch = input_ids_batch.to(self.device)
        attention_mask_batch = attention_mask_batch.to(self.device)
        
        return (input_ids_batch, attention_mask_batch)

    def inference(self, input_batch):
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []
        
        with torch.no_grad():
            predictions = self.model(input_ids_batch, token_type_ids=None, attention_mask=attention_mask_batch)
        logits = predictions[0]
        inferences.append(logits.argmax())

        return inferences

    def postprocess(self, inference_output):
        return inference_output

