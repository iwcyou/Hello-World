import math
import unittest
from collections import Counter
import numpy as np
import torch
from sgns import tokenize, positive_pairs, pair_array, SGNS, NegativeSampler, normalize
from analyze import retrieval_metrics, bm25_matrix, document_vectors


class CorrectnessTests(unittest.TestCase):
    def test_tokenization(self):
        self.assertEqual(tokenize("TP53, BRCA1 DNA-binding 123. {'@toggle': 'yes', '#text': 'Drosophila'}"),
                         ['tp53','brca1','dna-binding','drosophila'])

    def test_pairs_and_boundaries(self):
        actual = pair_array([[0,1,2],[3,4]],1)
        self.assertEqual(Counter(map(tuple,actual)),Counter([(0,1),(1,0),(1,2),(2,1),(3,4),(4,3)]))
        seq = [0,-1,2,3]
        self.assertEqual(Counter(map(tuple,pair_array([seq],2))),Counter(positive_pairs(seq,2)))
        self.assertNotIn((0,2),list(positive_pairs(seq,1)))

    def test_loss_against_scalar_and_bce(self):
        model = SGNS(4,2).double()
        with torch.no_grad():
            model.center.weight.copy_(torch.tensor([[.2,.3],[.1,-.4],[.6,.2],[-.3,.1]]))
            model.context.weight.copy_(torch.tensor([[.1,.5],[-.2,.4],[.3,-.1],[.7,.2]]))
        w,c,n = torch.tensor([0]),torch.tensor([1]),torch.tensor([[2,3]])
        v = model.center(w)[0]
        pos = float((v*model.context(c)[0]).sum().detach())
        neg = (v*model.context(n)[0]).sum(-1).detach().tolist()
        expected = math.log1p(math.exp(-pos))+sum(math.log1p(math.exp(x)) for x in neg)
        self.assertAlmostEqual(model(w,c,n).item(),expected,places=10)
        logits = torch.tensor([pos]+neg,dtype=torch.float64)
        reference = torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([1.,0.,0.],dtype=torch.float64),reduction='sum')
        self.assertAlmostEqual(model(w,c,n).item(),reference.item(),places=10)
        model(w,c,n).backward()
        self.assertTrue(torch.isfinite(model.center.weight.grad).all())
        self.assertGreater(model.center.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.context.weight.grad.abs().sum().item(),0)

    def test_sampler_conditional_distribution(self):
        torch.manual_seed(42)
        sampler = NegativeSampler([1,4,9,16])
        samples = sampler.sample(torch.zeros(20000,dtype=torch.long),5)
        self.assertFalse((samples==0).any())
        observed = torch.bincount(samples.flatten(),minlength=4)/samples.numel()
        expected = sampler.probabilities.clone();expected[0]=0;expected/=expected.sum()
        self.assertTrue(torch.allclose(observed,expected,atol=.01))

    def test_training_decreases_fixed_loss(self):
        torch.manual_seed(7)
        model = SGNS(4,8)
        optimizer = torch.optim.Adam(model.parameters(),lr=.05)
        w,c,n = torch.tensor([0,1]),torch.tensor([1,0]),torch.tensor([[2,3],[2,3]])
        initial = model(w,c,n).item()
        before = model.center.weight.detach().clone()
        for _ in range(30):
            optimizer.zero_grad();loss=model(w,c,n);loss.backward();optimizer.step()
        self.assertLess(model(w,c,n).item(),initial*.2)
        self.assertFalse(torch.equal(before,model.center.weight))

    def test_retrieval_metrics(self):
        values = retrieval_metrics([['a','b'],['b','a'],['b','c']],['a','a','a'])
        self.assertAlmostEqual(values['MRR@5'],.5)
        self.assertAlmostEqual(values['target_recall@5'],2/3)
        self.assertEqual(retrieval_metrics([['b']*5+['a']],['a'])['MRR@5'],0)

    def test_vector_oov_and_bm25(self):
        vectors = np.array([[1.,0.],[0.,1.]])
        dense, encode, _ = document_vectors([['gene'],['brain']],['gene','brain'],vectors)
        np.testing.assert_allclose(encode(['unknown']),[0,0])
        np.testing.assert_allclose(normalize(np.zeros((1,2))),[[0,0]])
        bm, vocab = bm25_matrix([['gene'],['brain']])
        self.assertGreater(bm[0,vocab['gene']],bm[1,vocab['gene']])

if __name__=='__main__':
    unittest.main()
