class CITADELRetrievalTask(MultiVecRetrieverTask):
    def __init__(
        self,
        ctx_embeddings_dir,
        checkpoint_path,
        index2docid_path=None,
        hnsw_index=False,
        output_path="/tmp/results.jsonl",
        passages="",
        topk=100,
        cuda=True,
        portion=1.0,
        quantizer=None,
        sub_vec_dim=4,
        expert_parallel=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctx_embeddings_dir = ctx_embeddings_dir
        self.checkpoint_path = checkpoint_path
        self.index2docid_path = index2docid_path
        self.hnsw_index = hnsw_index
        self.output_path = output_path
        self.passages = passages
        self.topk = topk
        self.cuda = cuda
        self.quantizer = quantizer if quantizer != "None" else None
        self.sub_vec_dim = sub_vec_dim
        self.portion = portion
        self.expert_parallel = expert_parallel
        self.latency = collections.defaultdict(float)