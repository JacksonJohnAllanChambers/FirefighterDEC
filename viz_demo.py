import time
import numpy as np
import matplotlib.pyplot as plt
import fire_viz


def main():
    # Use built-in dummy generators from fire_viz to quickly test the plotting
    hist, pred = fire_viz.generate_dummy_maps()
    drones = fire_viz.generate_dummy_drones(num=4)

    plt.ion()
    viz = fire_viz.FireViz(pred_alpha=0.4, wind_stride=12)  # coarser wind arrows for speed
    rng = np.random.default_rng(9)

    def crazy_spread(pred_map, round_idx):
        H, W = fire_viz.HEIGHT, fire_viz.WIDTH
        new_map = [row[:] for row in pred_map]

        # Random flare-ups: ignite some new spots
        for _ in range(40):
            x = int(rng.integers(0, W)); y = int(rng.integers(0, H))
            sev = int(rng.integers(5, 10))
            wind = int(rng.integers(3, 9)); wdir = int(rng.integers(0, 4))
            cit = 0; ff = 0
            new_map[y][x] = fire_viz.encode_tile(x, y, sev, wind, wdir, cit, ff, n=round_idx % 10, trust=7)

        # Broad spread from existing fires (8-neighborhood), with bias to wind dir
        for y in range(H):
            for x in range(W):
                s = pred_map[y][x]
                if s == '\\0':
                    continue
                t = fire_viz.parse_tile(s)
                if not t:
                    continue
                sev = int(t["fire"]) ; wind = int(t["windspeed"]) ; wdir = int(t["winddir"]) ; cit = t["citizen"]; ff = t["firefighter"]

                # Jitter severity locally
                sev_local = max(0, min(9, sev + int(rng.integers(-1, 2))))
                new_map[y][x] = fire_viz.encode_tile(x, y, sev_local, wind, wdir, cit, ff, n=round_idx % 10, trust=8)

                # Spread probability increases with wind and severity
                p_base = 0.08 + 0.04 * wind + 0.03 * sev
                p_base = float(max(0.05, min(0.92, p_base)))

                # Neighbor deltas (8-connected). Wind dir gets extra bias.
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < W and 0 <= ny < H):
                            continue
                        bias = 1.0
                        wx, wy = fire_viz.DIR_TO_VEC[wdir]
                        if (dx, dy) == (wx, wy):
                            bias = 1.6
                        elif dx == wx or dy == wy:
                            bias = 1.25
                        p = p_base * bias
                        if rng.random() < p:
                            add = int(rng.integers(1, 3))
                            new_sev = max(1, min(9, (sev // 2) + add))
                            # keep neighbor's wind/dir if already exists
                            if new_map[ny][nx] != '\\0':
                                pt = fire_viz.parse_tile(new_map[ny][nx])
                                if pt:
                                    ns = max(pt["fire"], new_sev)
                                    nw, nd = pt["windspeed"], pt["winddir"]
                                    new_map[ny][nx] = fire_viz.encode_tile(nx, ny, ns, nw, nd, max(cit, pt["citizen"]), max(ff, pt["firefighter"]), n=round_idx % 10, trust=6)
                                    continue
                            new_map[ny][nx] = fire_viz.encode_tile(nx, ny, new_sev, wind, wdir, cit, ff, n=round_idx % 10, trust=6)

        # Occasional cooling in a random stripe to add contrast
        if round_idx % 3 == 0:
            y0 = int(rng.integers(0, H))
            for x in range(W):
                s = new_map[y0][x]
                if s != '\\0':
                    t = fire_viz.parse_tile(s)
                    if t:
                        sev = max(0, t["fire"] - int(rng.integers(1, 3)))
                        new_map[y0][x] = fire_viz.encode_tile(x, y0, sev, t["windspeed"], t["winddir"], t["citizen"], t["firefighter"], n=round_idx % 10, trust=7)

        return new_map

    for round_idx in range(10):
        # Make the prediction layer change dramatically between frames
        pred = crazy_spread(pred, round_idx)

        # Move drones a bit for motion
        for d in drones:
            d.x = int(np.clip(d.x + int(rng.integers(-5, 6)), 0, fire_viz.WIDTH-1))
            d.y = int(np.clip(d.y + int(rng.integers(-3, 4)), 0, fire_viz.HEIGHT-1))

        viz.draw(
            hist, pred, drones,
            title=f"Visualization Demo — Round {round_idx}",
            show_scan_windows=True,
        )
    # Let the GUI event loop process and render this frame; avoid extra sleep
    plt.show(block=False)
    viz.fig.canvas.draw_idle()
    plt.pause(0.1)

    # Keep window open briefly at end
    time.sleep(2.5)


if __name__ == "__main__":
    main()
