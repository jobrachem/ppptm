import pytest
import numpy as np

from ppptm.score import tw_mv_score, vectorized_tw_mv_score

def test_tw_mv_score_es():
    y = np.array([0.5, 1.5, 2.5])
    dat = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([0.5, 0.5, 0.5])

    result = tw_mv_score(y, dat, mu, sigma, scoring_rule="es")

    assert result == pytest.approx(1.534264)

def test_tw_mv_score_vs():
    y = np.array([0.5, 1.5, 2.5])
    dat = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([0.5, 0.5, 0.5])

    result = tw_mv_score(y, dat, mu, sigma, scoring_rule="vs")

    assert result == pytest.approx(4.037822)

def test_tw_mv_score_mmds():
    y = np.array([0.5, 1.5, 2.5])
    dat = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([0.5, 0.5, 0.5])

    result = tw_mv_score(y, dat, mu, sigma, scoring_rule="mmds")

    assert result == pytest.approx(0.1942287)

def test_tw_mv_score_es_bigger():
    y = np.random.normal(loc=0.0, scale=1.0, size=(100,))
    mu = np.full_like(y, fill_value=1.0)
    sigma = np.full_like(mu, fill_value=1.0)
    
    dat1 = np.random.normal(loc=0.0, scale=1.0, size=(100,5))
    result1 = tw_mv_score(y, dat1, mu, sigma, scoring_rule="es")

    dat2 = np.random.normal(loc=-1.0, scale=1.0, size=(100,5))
    result2 = tw_mv_score(y, dat2, mu, sigma, scoring_rule="es")

    assert result1 < result2

def test_tw_mv_score_vectorized():

    y = np.random.normal(loc=0.0, scale=1.0, size=(3, 100))
    mu = np.full(shape=y.shape[-1], fill_value=1.0)
    sigma = np.full_like(mu, fill_value=1.0)
    
    dat1 = np.random.normal(loc=0.0, scale=1.0, size=(5,100))
    result1 = vectorized_tw_mv_score(y, dat1, mu, sigma, scoring_rule="es")

    dat2 = np.random.normal(loc=-1.0, scale=1.0, size=(5,100))
    result2 = vectorized_tw_mv_score(y, dat2, mu, sigma, scoring_rule="es")

    assert result1 < result2


