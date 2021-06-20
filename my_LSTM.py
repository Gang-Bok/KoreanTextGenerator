import cupy as cp
import time


def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


def softmax(x):
    exp_x = cp.exp(x)
    sum_exp_x = cp.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


class my_LSTM_batch:
    def __init__(self, hidden_layer, input_size, output_size):

        self.hidden_layer = hidden_layer
        self.input_size = input_size
        self.output_size = output_size

        self.E_x = cp.random.randn(input_size, output_size)

        self.W_xf = cp.random.randn(hidden_layer, input_size) * 0.1
        self.W_xi = cp.random.randn(hidden_layer, input_size) * 0.1
        self.W_xo = cp.random.randn(hidden_layer, input_size) * 0.1
        self.W_xg = cp.random.randn(hidden_layer, input_size) * 0.1

        self.W_hf = cp.random.randn(hidden_layer, hidden_layer) * 0.1
        self.W_hi = cp.random.randn(hidden_layer, hidden_layer) * 0.1
        self.W_ho = cp.random.randn(hidden_layer, hidden_layer) * 0.1
        self.W_hg = cp.random.randn(hidden_layer, hidden_layer) * 0.1

        self.W_y = cp.random.randn(output_size, hidden_layer) * 0.1

        self.b_f = cp.zeros((hidden_layer, 1)) + 1
        self.b_i = cp.zeros((hidden_layer, 1)) + 1
        self.b_o = cp.zeros((hidden_layer, 1)) + 1
        self.b_g = cp.zeros((hidden_layer, 1)) + 1

        self.b_y = cp.zeros((output_size, 1)) + 1

        self.h_prev = cp.zeros((hidden_layer, 1))
        self.c_prev = cp.zeros((hidden_layer, 1))

    def loss_function(self, x, y, h_prev, c_prev, start_idx, end_idx):
        f = {}
        i = {}
        o = {}
        g = {}
        h = {}
        c = {}
        y_hat = {}
        p = {}
        loss = 0

        y_true = {}
        one_hot = {}
        x_tmp = {}
        target_cnt = 0

        for k in range(start_idx, end_idx):
            # Embedding
            for idx2 in range(0, len(x[k])):
                idx = (k-start_idx)*len(x[k]) + idx2
                one_hot[idx] = cp.zeros((self.output_size, 1))
                one_hot[idx][x[k][idx2]] = 1
                x_tmp[idx] = cp.zeros((1, self.input_size))
                x_tmp[idx] = cp.dot(self.E_x, one_hot[idx])
            h[-1] = cp.copy(h_prev)
            c[-1] = cp.copy(c_prev)
            # 순전파
            for idx2 in range(0, len(x[k])):
                idx = (k-start_idx) * len(x[k]) + idx2
                f[idx] = sigmoid(self.W_xf.dot(x_tmp[idx]) + cp.dot(self.W_hf, h[idx-1]) + self.b_f)
                i[idx] = sigmoid(self.W_xi.dot(x_tmp[idx]) + cp.dot(self.W_hi, h[idx-1]) + self.b_i)
                g[idx] = cp.tanh(self.W_xg.dot(x_tmp[idx]) + cp.dot(self.W_hg, h[idx-1]) + self.b_g)
                o[idx] = sigmoid(self.W_xo.dot(x_tmp[idx]) + cp.dot(self.W_ho, h[idx-1]) + self.b_o)

                c[idx] = f[idx] * c[idx - 1] + i[idx] * g[idx]
                h[idx] = o[idx] * cp.tanh(c[idx])

            for idx2 in range(0, len(x[k])):
                idx = (k-start_idx) * len(x[k]) + idx2
                y_hat[idx] = cp.dot(self.W_y, h[idx]) + self.b_y
                p[idx] = softmax(y_hat[idx])
                y_true[idx] = cp.zeros((self.output_size, 1))
                if idx2 + 1 < len(x[k]):
                    if x[k][idx2] == 0:
                        continue
                    y_true[idx][x[k][idx2 + 1]] = 1
                    target_cnt += 1
                else:
                    y_true[idx] = y[k].reshape((y[k].shape[0], 1))
                    target_cnt += 1
                loss += -cp.log(cp.dot(p[idx].T, y_true[idx]))

        loss /= float(target_cnt)

        dW_xf = cp.zeros_like(self.W_xf)
        dW_xi = cp.zeros_like(self.W_xi)
        dW_xg = cp.zeros_like(self.W_xg)
        dW_xo = cp.zeros_like(self.W_xo)

        dW_hf = cp.zeros_like(self.W_hf)
        dW_hi = cp.zeros_like(self.W_hi)
        dW_hg = cp.zeros_like(self.W_hg)
        dW_ho = cp.zeros_like(self.W_ho)

        db_y = cp.zeros_like(self.b_y)

        db_f = cp.zeros_like(self.b_f)
        db_i = cp.zeros_like(self.b_i)
        db_g = cp.zeros_like(self.b_g)
        db_o = cp.zeros_like(self.b_o)

        dh_prev = cp.zeros_like(h[0])
        dc_prev = cp.zeros_like(c[0])

        dW_y = cp.zeros_like(self.W_y)

        dE_x = cp.zeros_like(self.E_x)
        # 역전파
        for k in range(start_idx, end_idx):
            for idx2 in range(len(x[k]) - 1, 0, -1):
                idx = (k-start_idx) * len(x[k]) + idx2
                if x[k][idx2] == 0:
                    break

                dy = p[idx]
                dy -= y_true[idx]
                dW_y += cp.dot(dy, h[idx].T)
                db_y += dy

                dh = cp.dot(self.W_y.T, dy) + dh_prev
                dc = (1 - cp.tanh(c[idx]) * cp.tanh(c[idx])) * dh * o[idx] + dc_prev
                dc_prev = f[idx] * dc

                df = c[idx - 1] * dc
                di = dc * g[idx]
                do = cp.tanh(c[idx]) * dh
                dg = dc * i[idx]

                dh_f = (1 - f[idx]) * f[idx] * df
                dh_i = (1 - i[idx]) * i[idx] * di
                dh_o = (1 - o[idx]) * o[idx] * do
                dh_g = (1 - g[idx] * g[idx]) * dg

                dW_xf += cp.dot(dh_f, x[k][idx2].T)
                dW_xi += cp.dot(dh_i, x[k][idx2].T)
                dW_xo += cp.dot(dh_o, x[k][idx2].T)
                dW_xg += cp.dot(dh_g, x[k][idx2].T)

                dW_hf += cp.dot(dh_f, h[idx-1].T)
                dW_hi += cp.dot(dh_i, h[idx-1].T)
                dW_ho += cp.dot(dh_o, h[idx-1].T)
                dW_hg += cp.dot(dh_g, h[idx-1].T)

                db_f += dh_f
                db_i += dh_i
                db_o += dh_o
                db_g += dh_g

                dh_prev = cp.dot(self.W_hf, dh_f) + cp.dot(self.W_hi, dh_i) + cp.dot(self.W_ho, dh_o) + cp.dot(self.W_hg, dh_g)
                dx = cp.dot(self.W_xf.T, dh_f) + cp.dot(self.W_xi.T, dh_i) + cp.dot(self.W_xo.T, dh_o) + cp.dot(self.W_xg.T, dh_g)
                dE_x = cp.dot(dx, one_hot[idx].T)

        for parm in [dW_xf, dW_xi, dW_xo, dW_xg, dW_hf, dW_hi, dW_ho, dW_hg, dW_y, db_f, db_i, db_o, db_g, db_y, dE_x]:
            cp.clip(parm, -5, 5, out=parm)

        return loss, dW_xf, dW_xi, dW_xo, dW_xg, \
               dW_hf, dW_hi, dW_ho, dW_hg, dW_y, \
               db_f, db_i, db_o, db_g, db_y, dE_x\


    def adam(self, m, s, d, t):
        epsilon = 0.00000001
        one = cp.ones_like(m)
        beta1 = cp.zeros_like(m) + 900
        beta2 = cp.zeros_like(s) + 990

        divmat = cp.zeros_like(m) + 1000

        beta1 = cp.divide(beta1, divmat)
        beta2 = cp.divide(beta2, divmat)

        learning_rate = cp.ones_like(m)
        learning_rate = cp.divide(learning_rate, divmat)

        div1 = (one - (beta1 ** t))
        div2 = (one - (beta2 ** t))

        m = (m * beta1) + (one - beta1) * d
        m_prime = cp.divide(m, div1)
        s = (s * beta2) + (one - beta2) * (d ** 2)
        s_prime = cp.divide(s, div2)

        result = cp.divide(learning_rate, cp.sqrt(s_prime)+epsilon) * m_prime
        return m, s, result

    def get_lr(self, d):
        learning_rate = cp.ones_like(d)
        divmat = cp.zeros_like(d) + 100
        learning_rate = cp.divide(learning_rate, divmat)
        return learning_rate

    def train_adam(self, x, y, epochs, batch_size):

        mdW_xf = cp.zeros_like(self.W_xf)
        mdW_xi = cp.zeros_like(self.W_xi)
        mdW_xo = cp.zeros_like(self.W_xo)
        mdW_xg = cp.zeros_like(self.W_xg)

        mdW_hf = cp.zeros_like(self.W_hf)
        mdW_hi = cp.zeros_like(self.W_hi)
        mdW_ho = cp.zeros_like(self.W_ho)
        mdW_hg = cp.zeros_like(self.W_hg)

        mdE_x = cp.zeros_like(self.E_x)

        mdW_y = cp.zeros_like(self.W_y)

        mdb_f = cp.zeros_like(self.b_f)
        mdb_i = cp.zeros_like(self.b_i)
        mdb_o = cp.zeros_like(self.b_o)
        mdb_g = cp.zeros_like(self.b_g)

        mdb_y = cp.zeros_like(self.b_y)

        sdW_xf = cp.zeros_like(self.W_xf)
        sdW_xi = cp.zeros_like(self.W_xi)
        sdW_xo = cp.zeros_like(self.W_xo)
        sdW_xg = cp.zeros_like(self.W_xg)

        sdW_hf = cp.zeros_like(self.W_hf)
        sdW_hi = cp.zeros_like(self.W_hi)
        sdW_ho = cp.zeros_like(self.W_ho)
        sdW_hg = cp.zeros_like(self.W_hg)

        sdE_x = cp.zeros_like(self.E_x)

        sdW_y = cp.zeros_like(self.W_y)

        sdb_f = cp.zeros_like(self.b_f)
        sdb_i = cp.zeros_like(self.b_i)
        sdb_o = cp.zeros_like(self.b_o)
        sdb_g = cp.zeros_like(self.b_g)

        sdb_y = cp.zeros_like(self.b_y)

        bef_loss = 2147483647
        exit_cnt = 0

        t = 1
        total_time = 0
        loss_history = []
        for i in range(0, epochs):
            loss_sum = 0
            print("-------------------------------------iter : %d Start -------------------------------- " % i)
            start = time.time()
            start_idx = 0
            end_idx = start_idx + batch_size
            batch_now = 0
            while start_idx < len(x):
                if end_idx > len(x):
                    end_idx = len(x)
                loss, dW_xf, dW_xi, dW_xo, dW_xg, \
                dW_hf, dW_hi, dW_ho, dW_hg, dW_y, \
                db_f, db_i, db_o, db_g, db_y, dE_x\
                = self.loss_function(x, y, self.h_prev, self.c_prev, start_idx, end_idx)

                start_idx += batch_size
                end_idx += batch_size

                loss_sum += loss

                # adam optimizer
                mdW_xf, sdW_xf, lr_W_xf = self.adam(mdW_xf, sdW_xf, dW_xf, t)
                mdW_xi, sdW_xi, lr_W_xi = self.adam(mdW_xi, sdW_xi, dW_xi, t)
                mdW_xo, sdW_xo, lr_W_xo = self.adam(mdW_xo, sdW_xo, dW_xo, t)
                mdW_xg, sdW_xg, lr_W_xg = self.adam(mdW_xg, sdW_xg, dW_xg, t)

                mdW_hf, sdW_hf, lr_W_hf = self.adam(mdW_hf, sdW_hf, dW_hf, t)
                mdW_hi, sdW_hi, lr_W_hi = self.adam(mdW_hi, sdW_hi, dW_hi, t)
                mdW_ho, sdW_ho, lr_W_ho = self.adam(mdW_ho, sdW_ho, dW_ho, t)
                mdW_hg, sdW_hg, lr_W_hg = self.adam(mdW_hg, sdW_hg, dW_hg, t)

                mdE_x, sdE_x, lr_E_x = self.adam(mdE_x, sdE_x, dE_x, t)

                mdW_y, sdW_y, lr_W_y = self.adam(mdW_y, sdW_y, dW_y, t)

                mdb_f, sdb_f, lr_b_f = self.adam(mdb_f, sdb_f, db_f, t)
                mdb_i, sdb_i, lr_b_i = self.adam(mdb_i, sdb_i, db_i, t)
                mdb_o, sdb_o, lr_b_o = self.adam(mdb_o, sdb_o, db_o, t)
                mdb_g, sdb_g, lr_b_g = self.adam(mdb_g, sdb_g, db_g, t)

                mdb_y, sdb_y, lr_b_y = self.adam(mdb_y, sdb_y, db_y, t)

                self.W_xf = self.W_xf - lr_W_xf
                self.W_xi = self.W_xi - lr_W_xi
                self.W_xo = self.W_xo - lr_W_xo
                self.W_xg = self.W_xg - lr_W_xg

                self.W_hf = self.W_hf - lr_W_hf
                self.W_hi = self.W_hi - lr_W_hi
                self.W_ho = self.W_ho - lr_W_ho
                self.W_hg = self.W_hg - lr_W_hg

                self.E_x = self.E_x - lr_E_x

                self.W_y = self.W_y - lr_W_y

                self.b_f = self.b_f - lr_b_f
                self.b_i = self.b_i - lr_b_i
                self.b_o = self.b_o - lr_b_o
                self.b_g = self.b_g - lr_b_g

                self.b_y = self.b_y - lr_b_y
                print("batch %d loss is %f" % (batch_now, loss))
                batch_now += 1

                t += 1
            recent_time = time.time() - start
            total_time += recent_time
            print("iter : %d complete --------------------- Take time : %f -------- Loss Avg: %f " % (i, recent_time, loss_sum / batch_now))
            loss_history.append(loss_sum[0][0].tolist() / batch_now)
            # self.save_weight()
            if bef_loss <= loss_sum:
                exit_cnt += 1
            bef_loss = loss_sum
            if exit_cnt == 50:
                print("성능이 나아지질 않아서 조기 종료합니다.")
                return
        print("Final Take time : %f" % total_time)
        l_h = cp.array(loss_history)
        cp.save('Loss_history', l_h)

    def train_gradient_decent(self, x, y, epochs):
        bef_loss = 2147483647
        exit_cnt = 0
        total_time = 0
        for i in range(1, epochs+1):
            print("-------------------------------------iter : %d Start -------------------------------- " % i)
            start = time.time()
            loss_sum = 0
            for j in range(0, len(x)):
                loss, dW_xf, dW_xi, dW_xo, dW_xg, \
                dW_hf, dW_hi, dW_ho, dW_hg, dW_y, \
                db_f, db_i, db_o, db_g, db_y, dE_x\
                = self.loss_function(x[j], y[j], self.h_prev, self.c_prev)
                loss_sum += loss
                # adam optimizer
                lr_W_xf = self.get_lr(dW_xf)
                lr_W_xi = self.get_lr(dW_xi)
                lr_W_xo = self.get_lr(dW_xo)
                lr_W_xg = self.get_lr(dW_xg)

                lr_W_hf = self.get_lr(dW_hf)
                lr_W_hi = self.get_lr(dW_hi)
                lr_W_ho = self.get_lr(dW_ho)
                lr_W_hg = self.get_lr(dW_hg)

                lr_E_x = self.get_lr(dE_x)

                lr_W_y = self.get_lr(dW_y)

                lr_b_f = self.get_lr(db_f)
                lr_b_i = self.get_lr(db_i)
                lr_b_o = self.get_lr(db_o)
                lr_b_g = self.get_lr(db_g)

                lr_b_y = self.get_lr(db_y)

                self.W_xf = self.W_xf - lr_W_xf * dW_xf
                self.W_xi = self.W_xi - lr_W_xi * dW_xi
                self.W_xo = self.W_xo - lr_W_xo * dW_xo
                self.W_xg = self.W_xg - lr_W_xg * dW_xg

                self.W_hf = self.W_hf - lr_W_hf * dW_hf
                self.W_hi = self.W_hi - lr_W_hi * dW_hi
                self.W_ho = self.W_ho - lr_W_ho * dW_ho
                self.W_hg = self.W_hg - lr_W_hg * dW_hg

                self.E_x = self.E_x - lr_E_x * dE_x

                self.W_y = self.W_y - lr_W_y * dW_y

                self.b_f = self.b_f - lr_b_f * db_f
                self.b_i = self.b_i - lr_b_i * db_i
                self.b_o = self.b_o - lr_b_o * db_o
                self.b_g = self.b_g - lr_b_g * db_g

                self.b_y = self.b_y - lr_b_y * db_y

                print("batch %d loss is %f" % (j, loss_sum / (j+1)))

            recent_time = time.time() - start
            total_time += recent_time
            print("iter : %d complete --------------------- Take time : %f Loss : %f" % (i, recent_time, loss_sum/len(x)))
            # self.save_weight()
            if bef_loss <= loss_sum:
                exit_cnt += 1
            bef_loss = loss_sum
            if exit_cnt == 50:
                print("성능이 나아지질 않아서 조기 종료합니다.")
                return
        print("Final Take time : %f" % total_time)

    def predict(self, x):
        f = {}
        i = {}
        o = {}
        g = {}
        h = {}
        c = {}
        y_hat = {}

        one_hot = {}
        x_tmp = {}

        # Embedding
        for idx in range(0, len(x[0])):
            one_hot[idx] = cp.zeros((self.output_size, 1))
            one_hot[idx][x[0][idx]] = 1
            x_tmp[idx] = cp.zeros((1, self.input_size))
            x_tmp[idx] = cp.dot(self.E_x, one_hot[idx])

        h[-1] = cp.copy(self.h_prev)
        c[-1] = cp.copy(self.c_prev)
        # 순전파
        for idx in range(0, len(x[0])):
            if x[0][idx] == 0:
                h[idx] = cp.copy(self.h_prev)
                c[idx] = cp.copy(self.c_prev)
                continue
            f[idx] = sigmoid(self.W_xf.dot(x_tmp[idx]) + cp.dot(self.W_hf, h[idx - 1]) + self.b_f)
            i[idx] = sigmoid(self.W_xi.dot(x_tmp[idx]) + cp.dot(self.W_hi, h[idx - 1]) + self.b_i)
            g[idx] = cp.tanh(self.W_xg.dot(x_tmp[idx]) + cp.dot(self.W_hg, h[idx - 1]) + self.b_g)
            o[idx] = sigmoid(self.W_xo.dot(x_tmp[idx]) + cp.dot(self.W_ho, h[idx - 1]) + self.b_o)

            c[idx] = f[idx] * c[idx - 1] + i[idx] * g[idx]
            h[idx] = o[idx] * cp.tanh(c[idx])

        y_hat[len(x[0]) - 1] = cp.dot(self.W_y, h[len(x[0]) - 1]) + self.b_y
        return y_hat[len(x[0]) - 1]

    def save_weight(self):
        cp.save('W_xf', self.W_xf)
        cp.save('W_xi', self.W_xi)
        cp.save('W_xo', self.W_xo)
        cp.save('W_xg', self.W_xg)

        cp.save('W_y', self.W_y)

        cp.save('E_x', self.E_x)

        cp.save('W_hf', self.W_hf)
        cp.save('W_hi', self.W_hi)
        cp.save('W_ho', self.W_ho)
        cp.save('W_hg', self.W_hg)

        cp.save('b_f', self.b_f)
        cp.save('b_i', self.b_i)
        cp.save('b_o', self.b_o)
        cp.save('b_g', self.b_g)

        cp.save('b_y', self.b_y)

    def load_weight(self):

        self.E_x = cp.load(r'test2_my\E_x.npy')

        self.W_xf = cp.load(r'test2_my\W_xf.npy')
        self.W_xi = cp.load(r'test2_my\W_xi.npy')
        self.W_xo = cp.load(r'test2_my\W_xo.npy')
        self.W_xg = cp.load(r'test2_my\W_xg.npy')

        self.W_hf = cp.load(r'test2_my\W_hf.npy')
        self.W_hi = cp.load(r'test2_my\W_hi.npy')
        self.W_ho = cp.load(r'test2_my\W_ho.npy')
        self.W_hg = cp.load(r'test2_my\W_hg.npy')

        self.b_f = cp.load(r'test2_my\b_f.npy')
        self.b_i = cp.load(r'test2_my\b_i.npy')
        self.b_o = cp.load(r'test2_my\b_o.npy')
        self.b_g = cp.load(r'test2_my\b_g.npy')

        self.W_y = cp.load(r'test2_my\W_y.npy')

        self.b_y = cp.load(r'test2_my\b_y.npy')
