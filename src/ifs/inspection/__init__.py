"""Prompt-injection inspection: `/v1/inspect`.

A sibling of `detectors/`, not a part of it. Detection asks whether an image
was generated; inspection asks whether it carries instructions meant for the
next model that reads it. Different question, different evidence, different
answer shape -- and no weights of its own, because the two models it needs are
served by sibling services.
"""
