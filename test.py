from infer import G2PInference

g2p = G2PInference("best_model.pt")
print(g2p("hello my name is roan and its nice to meet you"))