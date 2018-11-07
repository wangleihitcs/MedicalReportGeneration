from pycocoevalcap.meteor.meteor import Meteor

gts = {
    '1':['a good boy smiles'],
    '2':['a good boy smiles'],
}
res = {
    '1':['a good boy smiles'],
    '2':['a good boy smiles'],
}

meteor_scorer = Meteor()
meteor, _ = meteor_scorer.compute_score(gts=gts, res=res)
print(meteor)