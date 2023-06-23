import pickle
import re
import unicodedata

import torch
from lxml import etree
from lxml.html.clean import Cleaner
import lxml
from torch import nn


def clean_spaces(text):
    r"""Clean extra spaces in a string.

    Example:
      input: " asd  qwe   " --> output: "asd qwe"
      input: " asd\t qwe   " --> output: "asd qwe"
    Args:
      text: the input string with potentially extra spaces.

    Returns:
      a string containing only the necessary spaces.
    """
    return " ".join(re.split(r"\s+", text.strip()))


def clean_format_str(text):
    """Cleans unicode control symbols, non-ascii chars, and extra blanks."""
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = "".join([c if ord(c) < 128 else "" for c in text])
    text = clean_spaces(text)
    return text


def get_dom_tree(html, website):
    cleaner = Cleaner()
    cleaner.javascript = True
    cleaner.style = True
    html = html.replace("\0", "")  # Delete NULL bytes.
    # Replace the <br> tags with a special token for post-processing the xpaths.
    html = html.replace("<br>", "--BRRB--")
    html = html.replace("<br/>", "--BRRB--")
    html = html.replace("<br />", "--BRRB--")
    html = html.replace("<BR>", "--BRRB--")
    html = html.replace("<BR/>", "--BRRB--")
    html = html.replace("<BR />", "--BRRB--")

    # A special case in this website, where the values are inside the comments.
    if website == "careerbuilder":
        html = html.replace("<!--<tr>", "<tr>")
        html = html.replace("<!-- <tr>", "<tr>")
        html = html.replace("<!--  <tr>", "<tr>")
        html = html.replace("<!--   <tr>", "<tr>")
        html = html.replace("</tr>-->", "</tr>")

    html = clean_format_str(html)
    etree_root = cleaner.clean_html(lxml.html.fromstring(html))
    dom_tree = etree.ElementTree(etree_root)
    return dom_tree


def get_activation(activation):
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'ReLU6':
        return nn.ReLU6()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'CELU':
        return nn.CELU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU()
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'RReLU':
        return nn.RReLU()
    elif activation == 'GELU':
        return nn.GELU()
    elif activation == 'Sigmoid':
        return nn.Sigmoid()
    elif activation == 'Tanh':
        return nn.Tanh()
    elif activation == 'Hardtanh':
        return nn.Hardtanh()
    elif activation == 'LogSigmoid':
        return nn.LogSigmoid()
    elif activation == 'Softmax':
        return nn.Softmax()
    elif activation == 'Softmax2d':
        return nn.Softmax2d()
    elif activation == 'LogSoftmax':
        return nn.LogSoftmax()
    else:
        raise ValueError(
            "Invalid activation function. "
            "Choose from: ReLU, ReLU6, ELU, SELU, CELU, LeakyReLU, PReLU, RReLU, GELU, Sigmoid, Tanh, Hardtanh, "
            "LogSigmoid, Softmax, Softmax2d, LogSoftmax, "
            f"Now as: {activation}")


def load_from_pkl(path, mode='rb'):
    with open(path, mode) as fp:
        data = pickle.load(fp)
    return data