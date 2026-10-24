"""PICO pelvis twist relative to both feet -> G1-23 waist yaw (channel 12)."""
from collections import deque
import math
import statistics
from g1_pico_walk import valid_pose


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def unity_heading(pose):
    """Heading of local +Z projected into Unity's XZ plane; positive toward +X."""
    if not valid_pose(pose):
        raise ValueError('invalid body orientation')
    x, y, z, w = map(float, pose[3:])
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    fx, fz = 2*(x*z+y*w), 1-2*(x*x+y*y)
    if math.hypot(fx, fz) < 0.5:
        raise ValueError('body forward axis too tilted')
    return math.atan2(fx, fz)


def circular_mean(angles):
    sx, sz = sum(math.sin(v) for v in angles), sum(math.cos(v) for v in angles)
    if math.hypot(sx, sz)/len(angles) < 0.5:
        raise ValueError('ambiguous foot/body headings')
    return math.atan2(sx, sz)


def controller_position_in_pelvis_frame(controller_pose, pelvis_pose):
    """Translate/yaw-rotate controller position into the moving human pelvis frame."""
    if not valid_pose(controller_pose):
        raise ValueError('invalid controller pose')
    heading = unity_heading(pelvis_pose)
    x, y, z = (float(controller_pose[i])-float(pelvis_pose[i]) for i in range(3))
    c, s = math.cos(heading), math.sin(heading)
    return [c*x-s*z, y, s*x+c*z]


class WaistMapper:
    def __init__(self, max_degrees=10.0):
        if not 0 < max_degrees <= 100:
            raise ValueError('waist range must be within 100 degrees')
        self.limit = math.radians(max_degrees)
        self.generation = 0
        self.reset()

    def reset(self):
        self.requested = False
        self.samples = deque()
        self.baseline = None
        self.previous_frame = None
        self.previous_relative = None
        self.filtered = 0.0
        self.last_time = None
        self.last_issue = ''
        self.last_good_source = None
        self.rearm_required = False

    def begin_calibration(self):
        self.reset()
        self.generation += 1
        self.requested = True

    def update(self, packet, now):
        result = dict(valid=False, calibrated=self.baseline is not None, enabled=False,
                      generation=self.generation, robot_offset_rad=0.0)
        try:
            stamp = float(packet['body_source_ts'])
            if not (0 <= now-stamp <= 0.25 and 0 <= now-float(packet['ts']) <= 0.25):
                raise ValueError('body data stale')
            if packet.get('tracking_valid') is not True:
                raise ValueError('tracking/focus unavailable')
            body = packet['body_joints']
            if len(body) != 24:
                raise ValueError('expected 24 Body joints')
            pelvis, left, right = (body[i] for i in (0, 7, 8))
            hp, hl, hr = map(unity_heading, (pelvis, left, right))
            if self.baseline is not None and self.last_good_source is not None:
                gap = stamp-self.last_good_source
                if gap > 0.75:
                    raise ValueError('body tracking interruption exceeded 0.75s')
                if gap > 0.25:
                    self.rearm_required = True
            self.last_good_source = stamp
            feet = circular_mean((hl, hr))
            relative = wrap(hp-feet)
            foot_height_difference = float(left[1])-float(right[1])
            foot_heading_difference = wrap(hl-hr)
            result.update(pelvis_deg=math.degrees(hp), feet_deg=math.degrees(feet),
                          relative_deg=math.degrees(relative))
            fresh_frame = stamp != self.previous_frame
            self.previous_frame = stamp
            right_grip = float(packet.get('r_grip', 0.0))
            if not math.isfinite(right_grip) or not 0 <= right_grip <= 1.01:
                raise ValueError('invalid right Grip')
            if self.baseline is None:
                if not self.requested:
                    result['reason'] = ((self.last_issue+'; ') if self.last_issue else '')+'stand square, release right Grip, press X to calibrate'
                    return result
                if right_grip >= 0.6:
                    self.samples.clear()
                    result['reason'] = 'release right Grip during calibration'
                    return result
                if fresh_frame:
                    self.samples.append((now, relative, foot_height_difference, foot_heading_difference))
                while self.samples and now-self.samples[0][0] > 1.5:
                    self.samples.popleft()
                if len(self.samples) < 100 or now-self.samples[0][0] < 1.0:
                    result['reason'] = 'calibrating: stand square and still for 1 second'
                    return result
                ref = self.samples[0][1]
                angles = [wrap(v[1]-ref) for v in self.samples]
                if max(angles)-min(angles) > math.radians(3):
                    result['reason'] = 'calibration moving; keep still'
                    return result
                self.baseline = (circular_mean([v[1] for v in self.samples]),
                                 statistics.median(v[2] for v in self.samples),
                                 circular_mean([v[3] for v in self.samples]))
                self.last_time = now
            angle0, foot_y0, foot_yaw0 = self.baseline
            result['calibrated'] = True
            if abs(foot_height_difference-foot_y0) > 0.05:
                result['reason'] = 'foot lifted: hold waist'
                return result
            if abs(wrap(foot_heading_difference-foot_yaw0)) > math.radians(35):
                result['reason'] = 'feet changed stance: hold waist/recalibrate'
                return result
            if fresh_frame:
                if self.previous_relative is not None and abs(wrap(relative-self.previous_relative)) > math.radians(25):
                    raise ValueError('body orientation jumped')
                self.previous_relative = relative
                delta = wrap(relative-angle0)
                dt = min(0.1, max(0.0, now-(self.last_time or now)))
                self.filtered += (1-math.exp(-dt/0.12))*wrap(delta-self.filtered)
                self.filtered = wrap(self.filtered)
                self.last_time = now
            delta = self.filtered
            # Installed PICO Body frame: onsite paired test requires the same yaw sign.
            offset = math.copysign(max(0.0, abs(delta)-math.radians(1.5)), delta)
            offset = min(self.limit, max(-self.limit, offset))
            if right_grip < 0.6:
                self.rearm_required = False
            result.update(valid=True, enabled=right_grip >= 0.6 and not packet.get('btn_b') and not self.rearm_required,
                          human_twist_deg=math.degrees(delta), robot_offset_rad=offset,
                          robot_offset_deg=math.degrees(offset), reason=(
                              'tracking recovered: release right Grip, then re-grip' if self.rearm_required else 'waist ready'))
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            transient = str(exc) in ('body data stale', 'tracking/focus unavailable', "'body_source_ts'", 'expected 24 Body joints')
            if (transient and self.baseline is not None and self.last_good_source is not None
                    and 0 <= now-self.last_good_source <= 0.75):
                self.rearm_required = True
                self.last_issue = str(exc)
                result.update(calibrated=True, reason=str(exc)+'; hold waist, release/re-grip after recovery')
                return result
            self.reset()
            self.last_issue = str(exc)
            result.update(calibrated=False, reason=str(exc)+'; X recalibration required')
        return result


class WaistLimiter:
    def __init__(self, measured_q, max_degrees=10.0, rate_degrees=10.0):
        if not 0 < max_degrees <= 100 or not 0 < rate_degrees <= 10:
            raise ValueError('waist range/rate exceeds validated command envelope')
        if not math.isfinite(measured_q) or abs(measured_q) > 2.5:
            raise ValueError('waist measured angle outside startup envelope')
        self.limit, self.rate = math.radians(max_degrees), math.radians(rate_degrees)
        self.anchor, self.output = measured_q, measured_q
        self.session_center = measured_q
        self.was_enabled = False

    def calibrate(self, measured_q):
        if not math.isfinite(measured_q) or abs(measured_q) > 2.5:
            raise ValueError('invalid measured waist angle')
        self.anchor = self.output = measured_q
        self.was_enabled = False

    def step(self, offset, enabled, measured_q, dt):
        if not math.isfinite(measured_q) or not math.isfinite(offset):
            raise ValueError('non-finite waist input')
        if not enabled:
            if self.was_enabled:
                self.output = measured_q  # Release/timeout stops residual target interpolation.
            self.was_enabled = False
            return self.output
        self.was_enabled = True
        target = self.anchor+max(-self.limit, min(self.limit, offset))
        # Recalibration must not ratchet the torso farther after every tracking reset.
        target = max(self.session_center-self.limit, min(self.session_center+self.limit, target))
        target = max(-2.5, min(2.5, target))
        step = self.rate*min(0.02, max(0.0, dt))
        self.output += min(step, max(-step, target-self.output))
        return self.output
