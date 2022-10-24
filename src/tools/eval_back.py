from mindspore.train.callback import Callback


class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, src_url, train_url, total_epochs, args, save_freq=50):
        super(EvaluateCallBack, self).__init__()
        self.args = args
        self.model = model
        self.eval_dataset = eval_dataset
        self.src_url = src_url
        self.train_url = train_url
        self.total_epochs = total_epochs
        self.save_freq = save_freq
        self.eval_while_train = args.eval_while_train
        self.best_mpck = 0.

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        if (cb_params.cur_epoch_num > self.total_epochs * 0.9 or int(
                cb_params.cur_epoch_num - 1) % 20 == 0 or cb_params.cur_epoch_num < 5) and self.eval_while_train:
            scores = self.model.eval(self.eval_dataset)
            mPCK, mPCKh = scores['PCK']
            print("================\n", f"mPCK: {mPCK} mPCKh: {mPCKh}", "================", flush=True)
            if mPCK > self.best_mpck:
                print(f"=> best_mpck: {mPCK}")
            self.best_mpck = mPCK
        if self.args.run_modelarts:
            import moxing as mox
            if cur_epoch_num % self.save_freq == 0:
                mox.file.copy_parallel(src_url=self.src_url, dst_url=self.train_url)
