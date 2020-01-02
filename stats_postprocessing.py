# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
public functions
"""


def transfer_frames_to_relative_time(_frame_step, _fps):
    if _frame_step is -1:
        return ''
    _seconds_iter = _frame_step // _fps
    m, s = divmod(_seconds_iter, 60)
    h, m = divmod(m, 60)
    return '%02d:%02d:%02d' % (h, m, s)  # Compatible with both python2 and python3
    # return f"%02d:%02d:%02d" % (h, m, s)


# stats_count unused
class stats_count(object):
    def __init__(self, threshold=5, gap=2):
        self.count = 0
        self.threshold = threshold
        self.gap = gap
        self.now_stats = False

    def trigger(self, trigger=True):
        if trigger:
            if self.count < self.threshold + self.gap:
                self.count += 1
            else:
                self.now_stats = True
        else:
            if self.count > self.threshold - self.gap:
                self.count -= 1
            else:
                self.now_stats = False

    def this_stats(self):
        return self.now_stats

    def trigger_stats(self, trigger=True):
        self.trigger(trigger=trigger)
        return self.this_stats()


class staff_station(object):
    def __init__(self, fps=30, video_name='', is_diff_phase=True, initial_phase=3):
        self.initial_phase = initial_phase
        self.fps = fps
        self.last_stats = False
        self.video_name = video_name
        self.is_diff_phase = is_diff_phase
        self.leaves_st = -1
        self.leaves_ed = -1
        self.on_job_st = -1
        self.on_job_ed = -1
        self.time_list = []

    def update_state(self, on_stats=False, frame_step=0):

        # 1/2 seconds may be long, three frames instead
        if frame_step < self.initial_phase:
            self.last_stats = on_stats
            return

        # staff enter
        if on_stats and (not self.last_stats):
            self.leaves_ed = frame_step
            self.on_job_st = (frame_step + self.fps) if self.is_diff_phase else frame_step
            self.on_job_ed = -1

        # staff leaves
        if (not on_stats) and self.last_stats:
            self.on_job_ed = frame_step
            self.time_list.append((self.leaves_st, self.leaves_ed, self.on_job_st, self.on_job_ed))
            self.leaves_st = (frame_step + self.fps) if self.is_diff_phase else frame_step
            self.leaves_ed = -1
            self.on_job_st = -1
            self.on_job_ed = -1
        self.last_stats = on_stats

    def get_time_sheet(self):
        self.time_list.append((self.leaves_st, self.leaves_ed, self.on_job_st, self.on_job_ed))
        on_job_st_list = []
        on_job_ed_list = []
        leave_st_list = []
        leave_ed_list = []
        for e in self.time_list:
            leave_st_list.append(transfer_frames_to_relative_time(e[0], self.fps))
            leave_ed_list.append(transfer_frames_to_relative_time(e[1], self.fps))
            on_job_st_list.append(transfer_frames_to_relative_time(e[2], self.fps))
            on_job_ed_list.append(transfer_frames_to_relative_time(e[3], self.fps))

        tmp_df = pd.DataFrame({'视频编号': [self.video_name for i in range(len(on_job_st_list))],
                               '在岗开始时间': on_job_st_list,
                               '在岗结束时间': on_job_ed_list,
                               '离岗开始时间': leave_st_list,
                               '离岗结束时间': leave_ed_list})
        return tmp_df


class job_station(object):
    def __init__(self, fps=30, _dir=None, video_name='', initial_phase=3, max_len=100000):
        self.video_name = video_name
        self.initial_phase = initial_phase
        self.max_len = max_len
        self.fps = fps
        self.job_st = -1
        self.job_ed = -1
        # this below three value unused
        # self.shot_image = None
        # self.shot_image_name = None
        # self.dir = _dir
        self.job_list = []
        self.last_stats = False

    def update_state(self, on_stats=False, frame_step=0):

        # 1/2 seconds
        if frame_step < self.fps * 0.5:
            # if on_stats and (not self.last_stats) and (self.shot_image is None):
            #    self.shot_image = shot_image
            #    self.shot_image_name = '{}-{}'.format(self.video_name, frame_step)
            self.last_stats = on_stats
            return

        # client enter
        if on_stats and (not self.last_stats):
            self.job_st = frame_step
            self.job_ed = -1

        # after client enter 3 seconds, we shot
        # if on_stats and self.job_st > 0 and (frame_step - self.job_st) > 3 * self.fps and (self.shot_image is None):
        #     self.shot_image = shot_image
        #     self.shot_image_name = '{}-{}'.format(self.video_name, frame_step)

        # staff leaves
        if (not on_stats) and self.last_stats:
            self.job_ed = frame_step
            job_id = len(self.job_list) + 1
            self.job_list.append((job_id, self.job_st, self.job_ed))
            self.job_st = -1
            self.job_ed = -1
            # self.shot_image = None
            # self.shot_image_name = None

        self.last_stats = on_stats

    def get_time_list(self):

        if self.job_st > 0:
            # if self.shot_image_name is None:
            #     self.shot_image_name = 'not capture'
            job_id = len(self.job_list) + 1
            self.job_list.append((job_id, self.job_st, self.job_ed))

        job_id_list = []
        job_st_list = []
        job_ed_list = []
        # job_shot_list = []
        for e in self.job_list:
            job_id_list.append(e[0])
            job_st_list.append(transfer_frames_to_relative_time(e[1], self.fps))
            job_ed_list.append(transfer_frames_to_relative_time(e[2], self.fps))
            # job_shot_list.append(e[3])

        job_tmp_df = pd.DataFrame({'视频编号': [self.video_name for _ in range(len(job_id_list))],
                                   '客户编号': job_id_list,
                                   '开始时间': job_st_list,
                                   '结束时间': job_ed_list})

        # fix bugs: when len(job_tmp_df), raise no keys in get_shot_step
        if len(job_tmp_df) == 0:
            return job_tmp_df

        def get_shot_step(_row):
            def get_time(str_time):
                h, m, s = map(int, str_time.split(':'))
                return h * 3600 + m * 60 + s

            t1 = get_time(_row['开始时间']) if len(_row['开始时间']) > 4 else 0
            t2 = get_time(_row['结束时间']) if len(_row['结束时间']) > 4 else self.max_len
            return np.int((t1 + t2) // 2 * self.fps)

        job_tmp_df['frame_step'] = job_tmp_df.apply(lambda row: get_shot_step(row), axis=1)
        job_tmp_df['客户截图'] = job_tmp_df.apply(lambda row: '%s-%d.jpg' % (row['视频编号'], row['frame_step']), axis=1)
        return job_tmp_df


class device_station(object):
    def __init__(self, fps=30, video_name='', is_diff_phase=False, initial_phase=3):
        self.initial_phase = initial_phase
        self.fps = fps
        self.last_stats = False
        self.video_name = video_name
        self.is_diff_phase = is_diff_phase
        self.leaves_st = -1
        self.leaves_ed = -1
        # split
        self.on_job_st = -1
        self.on_job_ed = -1
        self.idle_list = []
        self.on_job_list = []

    def update_state(self, on_stats=False, frame_step=0):

        # 1/2 seconds
        if frame_step < self.initial_phase:
            if self.last_stats and on_stats and self.on_job_st < 0:
                self.on_job_st = 0

            if (not self.last_stats) and (not on_stats) and self.leaves_st < 0:
                self.leaves_st = 0
            self.last_stats = on_stats
            return

        # people enter
        if on_stats and (not self.last_stats):
            self.leaves_ed = frame_step
            self.idle_list.append((self.leaves_st, self.leaves_ed))
            self.on_job_st = frame_step
            self.on_job_ed = -1
            self.leaves_st = -1
            self.leaves_ed = -1

        # people leaves
        if (not on_stats) and self.last_stats:
            self.on_job_ed = frame_step
            self.on_job_list.append((self.on_job_st, self.on_job_ed))
            self.leaves_st = frame_step
            self.leaves_ed = -1
            self.on_job_st = -1
            self.on_job_ed = -1

        self.last_stats = on_stats

    def get_idle_time_str(self):
        return ','.join(['{}-{}'.format(transfer_frames_to_relative_time(e[0], self.fps),
                                        transfer_frames_to_relative_time(e[1], self.fps))
                         for e in self.idle_list])

    def get_time_sheet(self, max_step=100000):
        if self.on_job_st > 0:
            self.on_job_list.append((self.on_job_st, max_step))
        if self.leaves_st > 0:
            self.idle_list.append((self.leaves_st, max_step))

        person_cnt = len(self.on_job_list)
        job_mean_minutes = sum([e[1] - e[0] for e in self.on_job_list]) / self.fps / 60 / person_cnt
        print(self.get_idle_time_str())
        return pd.DataFrame({'视频编号': [self.video_name],
                             '服务客户总数': person_cnt,
                             '平均服务时长': job_mean_minutes,
                             '设备空闲时间段': self.get_idle_time_str()})


def fill_hole_in_exist(tmp_df, limit=60, ignore_true_limit=0):
    tmp_df['exist_bak'] = tmp_df['exist']
    false_idx = tmp_df[not tmp_df['exist_bak']].index
    tmp_df.loc[false_idx, 'exist_bak'] = None
    tmp_df.fillna(method='pad', limit=limit, inplace=True)
    tmp_df['exist_bak_bak'] = None
    nan_idx = tmp_df[tmp_df['exist_bak'].isnull()].index
    tmp_df.loc[nan_idx, 'exist_bak_bak'] = False
    tmp_df.fillna(method='bfill', limit=limit, inplace=True)
    tmp_df.loc[tmp_df['exist_bak_bak'].isnull(), 'exist_bak_bak'] = True
    return tmp_df


def postprocessing_staff_from_pandas(tmp_df, fps=30, video_name='hello', limit=2, is_plot=False, is_diff_phase=False):
    _tmp_df = fill_hole_in_exist(tmp_df, limit=np.int(fps * limit))
    if is_plot:
        plt.plot(_tmp_df['iou'])
        plt.plot(_tmp_df['exist'])
        plt.plot(_tmp_df['exist_bak_bak'])
        plt.savefig(video_name + '.png')
        plt.close()
    staff = staff_station(fps=30, video_name=video_name, is_diff_phase=is_diff_phase)
    for i, row in _tmp_df.iterrows():
        staff.update_state(on_stats=row['exist_bak_bak'], frame_step=row['step'])
    _staff_df = staff.get_time_sheet()
    return _staff_df


def postprocessing_device_from_pandas(tmp_df, fps=30, video_name='hello', limit=2, is_plot=False):
    _tmp_df = fill_hole_in_exist(tmp_df, limit=np.int(fps * limit))
    if is_plot:
        plt.plot(_tmp_df['iou'])
        plt.plot(_tmp_df['exist'])
        plt.plot(_tmp_df['exist_bak_bak'])
        plt.savefig(video_name + '.png')
        plt.close()
    device = device_station(fps=30, video_name=video_name)
    for i, row in _tmp_df.iterrows():
        device.update_state(on_stats=row['exist_bak_bak'], frame_step=row['step'])
    _device_df = device.get_time_sheet(len(_tmp_df))
    return _device_df


def postprocessing_jobs_from_pandas(tmp_df, fps=30, video_name='hello', limit=2, is_plot=False):
    _tmp_df = fill_hole_in_exist(tmp_df, limit=np.int(fps * limit))
    if is_plot:
        plt.plot(_tmp_df['iou'])
        plt.plot(_tmp_df['exist'])
        plt.plot(_tmp_df['exist_bak_bak'])
        plt.savefig(video_name + '.png')
        plt.close()

    job = job_station(fps=30, video_name=video_name)
    for i, row in _tmp_df.iterrows():
        job.update_state(on_stats=row['exist_bak_bak'], frame_step=i)
    slave_tmp_df = job.get_time_list()
    return slave_tmp_df


def main_test_staff():
    _low, _high, _fps, _size = 800, 4000, 30, 11
    time_lg = np.int16(np.random.uniform(low=_low, high=_high, size=_size))
    on_job_list = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    print(time_lg / 30)
    frame_list = np.concatenate([np.ones(f) * e for e, f in zip(on_job_list, time_lg)])

    staff = staff_station(fps=30, video_name='hello', is_diff_phase=True)
    for i, e in enumerate(frame_list):
        staff.update_state(on_stats=e > 0.5, frame_step=i)
    tmp_df = staff.get_time_sheet()
    print(tmp_df)


def main_test_staff_v2():
    _fps = 30
    stats_cnt = stats_count(threshold=5, gap=1)
    # _low, _high, _fps, _size = 800, 4000, 30, 11
    # time_lg = np.int16(np.random.uniform(low=_low, high=_high, size=_size))
    # on_job_list = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    # print(time_lg / 30)
    # frame_list = np.concatenate([np.ones(f) * e for e, f in zip(on_job_list, time_lg)])
    tmp_df = pd.read_csv('/Users/jiangyy/projects/facenet/IMG_7438.MOV_step.csv')
    staff = staff_station(fps=30, video_name='hello', is_diff_phase=True)
    for i, row in tmp_df.iterrows():
        filter_exist = stats_cnt.trigger_stats(row['exist'])
        print(row['exist'], filter_exist)
        staff.update_state(on_stats=filter_exist, frame_step=row['step'])
    tmp_df = staff.get_time_sheet()
    print(tmp_df)


def main_test_jobs():
    df = pd.DataFrame({})
    on_job_list = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    _low, _high, _fps, _size = 800, 4000, 30, len(on_job_list)
    time_lg = np.int16(np.random.uniform(low=_low, high=_high, size=_size))
    print(np.int16(np.cumsum(time_lg / 30)) / 60)
    print(on_job_list)
    frame_list = np.concatenate([np.ones(f) * e for e, f in zip(on_job_list, time_lg)])
    job = job_station(fps=30, video_name='hello')
    for i, e in enumerate(frame_list):
        job.update_state(on_stats=e > 0.5, frame_step=i)
    slave_tmp_df = job.get_time_list()
    df = df.append(slave_tmp_df)

    on_job_list = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    _low, _high, _fps, _size = 800, 4000, 30, len(on_job_list)
    time_lg = np.int16(np.random.uniform(low=_low, high=_high, size=_size))
    print(np.int16(np.cumsum(time_lg / 30)) / 60)
    frame_list = np.concatenate([np.ones(f) * e for e, f in zip(on_job_list, time_lg)])
    job = job_station(fps=30, video_name='world')
    for i, e in enumerate(frame_list):
        job.update_state(on_stats=e > 0.5, frame_step=i)
    slave_tmp_df = job.get_time_list()
    df = df.append(slave_tmp_df)

    master_tmp_df = df.groupby('视频编号').size().reset_index().rename({0: '服务客户总数'}, axis=1)
    print(master_tmp_df)
    print(df)


def main_test_devices():
    on_job_list = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    _low, _high, _fps, _size = 800, 4000, 30, len(on_job_list)
    time_lg = np.int16(np.random.uniform(low=_low, high=_high, size=_size))
    print(np.int16(np.cumsum(time_lg / 30)) / 60)
    print(on_job_list)
    frame_list = np.concatenate([np.ones(f) * e for e, f in zip(on_job_list, time_lg)])
    device = device_station(fps=30, video_name='hello')
    for i, e in enumerate(frame_list):
        device.update_state(on_stats=e > 0.5, frame_step=i)
    tmp_df = device.get_time_sheet(i)
    print(tmp_df)


if __name__ == '__main__':
    # main_test_staff_v2()
    job_df = postprocessing_jobs_from_pandas(pd.read_csv('IMG_7481_step.csv'), fps=30)
    print(job_df)
    print(job_df.dtypes)
    # print(staff_df)
    # main_test_jobs()
