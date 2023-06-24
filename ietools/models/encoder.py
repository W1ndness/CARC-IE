import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers import MarkupLMConfig
from d2l import torch as d2l


devices = d2l.try_all_gpus()


class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-cased'):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(devices[0])

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(devices[0])
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def freeze(self, st, ed):
        st = max(0, st)
        ed = min(len(self.model.encoder.layer), ed)
        for layer in self.model.encoder.layer[st:ed]:
            for param in layer.parameters():
                param.requires_grad = False


class XPathEmbeddings(nn.Module):
    def __init__(self, config: MarkupLMConfig):
        super(XPathEmbeddings, self).__init__()
        self.max_depth = config.max_depth

        self.xpath_unitseq2_embeddings = nn.Linear(
            config.xpath_unit_hidden_size * self.max_depth, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.activation = nn.GELU()
        self.xpath_unitseq2_inner = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, 4 * config.hidden_size)
        self.inner2emb = nn.Linear(4 * config.hidden_size, config.hidden_size)

        self.xpath_tag_sub_embeddings = nn.ModuleList(
            [nn.Embedding(config.max_xpath_tag_unit_embeddings, config.xpath_unit_hidden_size) for _ in
             range(self.max_depth)])

        self.xpath_subs_sub_embeddings = nn.ModuleList(
            [nn.Embedding(config.max_xpath_subs_unit_embeddings, config.xpath_unit_hidden_size) for _ in
             range(self.max_depth)])

    def forward(self,
                xpath_tags_seq=None,
                xpath_subs_seq=None):
        xpath_tags_embeddings = [self.xpath_tag_sub_embeddings[i](xpath_tags_seq[:, :, i])
                                 for i in range(self.max_depth)]
        xpath_subs_embeddings = [self.xpath_subs_sub_embeddings[i](xpath_subs_seq[:, :, i])
                                 for i in range(self.max_depth)]

        xpath_tags_embeddings = torch.cat(xpath_tags_embeddings, dim=-1)
        xpath_subs_embeddings = torch.cat(xpath_subs_embeddings, dim=-1)

        xpath_embeddings = xpath_tags_embeddings + xpath_subs_embeddings

        xpath_embeddings = self.inner2emb(
            self.dropout(self.activation(self.xpath_unitseq2_inner(xpath_embeddings))))

        return xpath_embeddings

