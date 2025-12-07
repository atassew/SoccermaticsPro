# %%
#%pip install tqdm

# %%
from mplsoccer import Sbopen,Pitch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

# %%
parser = Sbopen()

# %%
# Define competitions and seasons
competitions = [
    
    {"competition_id": 1267, "season_id": 107, "name": "African Cup of Nations"}
    
]


# %%
matches = parser.match(1267, 107)
EG_games=matches[matches['home_team_name'] == 'Equatorial Guinea']['match_id'].tolist() + matches[matches['away_team_name'] == 'Equatorial Guinea']['match_id'].tolist()

# %%
#matches[(matches["home_team_name"] == 'Equatorial Guinea') | (matches["away_team_name"]=='Equatorial Guinea')][["home_team_name","away_team_name"]]



# %%
match_ids =[3922239, 3920408, 3920397, 3920386]
events_list = [parser.event(mid)[0] for mid in match_ids]
all_events = pd.concat(events_list, ignore_index=True)


# %%
nsue_shots = all_events[
    (all_events["player_name"] == "Emilio Nsue López") &
    (all_events["type_name"] == "Shot")
]
#nsue_shots["outcome_name"].value_counts()

# %%
total_shots = len(nsue_shots)

on_target = nsue_shots[
    nsue_shots["outcome_name"].isin(["Saved", "Goal", "Saved To Post"])
]

on_target_count = len(on_target)


# %%
shot_accuracy = (on_target_count / total_shots) * 100
print(f"Shot Accuracy: {shot_accuracy:.2f}%")

# %%

pitch=Pitch(line_color='black')
fig,ax=pitch.draw(figsize=(10,7))
for i,thepass in nsue_shots.iterrows():
    
        x=thepass['x']
        y=thepass['y']
        if thepass['outcome_name']=='Goal':
            color='red'
        else:
               color='blue'
        circleSize=thepass["shot_statsbomb_xg"]*5
        passCircle=plt.Circle((x,y),circleSize,color="blue")
        
        passCircle.set_alpha(0.2)
        ax.add_patch(passCircle)
        dx=thepass['end_x']-x
        dy=thepass['end_y']-y
        passArrow=plt.Arrow(x,y,dx,dy,width=3,color=color)
        ax.add_patch(passArrow)
ax.set_title("Emilio Nsue López shots in AFCON23",fontsize=24)
fig.set_size_inches(10,7)
plt.show()

# %%
#all_events["type_name"].unique()

# %%
player_name = "Emilio Nsue López"
passes_to_player = all_events[
    (all_events['type_name'] == 'Pass') &
    (all_events['pass_recipient_name'] == player_name)
]

# %%
passes_inside_box = passes_to_player[
    (passes_to_player['end_x'] >= 96) & (passes_to_player['end_x'] <= 120) &
    (passes_to_player['end_y'] >= 18) & (passes_to_player['end_y'] <= 62)
]

# %%
pass_ids = passes_inside_box['id'].tolist()


all_events_sorted = all_events.sort_values(['match_id', 'index']).reset_index(drop=True)


pass_event_rows = all_events_sorted[all_events_sorted['id'].isin(pass_ids)].index

receiver_actions = []

for row in pass_event_rows:
    pass_event = all_events_sorted.iloc[row]
    match_id = pass_event["match_id"]

    # search forward for Nsue's next event within the same match
    for next_row in range(row + 1, len(all_events_sorted)):
        next_event = all_events_sorted.iloc[next_row]

        # if match changes → stop
        if next_event["match_id"] != match_id:
            break

        # if Nsue performs the event → this is his next action
        if next_event["player_name"] == player_name:
            receiver_actions.append(next_event)
            break
# Convert to DataFrame
receiver_actions_df = pd.DataFrame(receiver_actions)
nsue_under_pressure_receipts = receiver_actions_df[
    receiver_actions_df['under_pressure'] == True
]
passes_per_match = nsue_under_pressure_receipts.groupby("match_id").size()
average_per_match = passes_per_match.mean()


print("Passes received under pressure inside box per match:")
print(passes_per_match)

print("\nAverage per match:", average_per_match)

# %%
# 1. Filter receptions under pressure
under_pressure_receptions = passes_to_player[
    passes_to_player['under_pressure'] == True
]

# 2. Sort these receptions to preserve match timeline
under_pressure_passes = under_pressure_receptions.sort_values(
    ['match_id', 'period', 'minute', 'second', 'index']
).copy()

kept_flags = []

for idx, row in under_pressure_passes.iterrows():

    # 3. Extract all events from this match
    match_events = all_events[
        all_events['match_id'] == row['match_id']
    ].sort_values(
        ['period', 'minute', 'second', 'index']
    ).reset_index(drop=False)  # "index" now contains StatsBomb index

    # 4. Real StatsBomb index for this reception event
    sb_index = row['index']

    # 5. Locate this event
    loc = match_events.index[match_events['index'] == sb_index]

    if len(loc) == 0:
        kept_flags.append(False)
        continue

    event_idx = loc[0]

    # ------------------------------------------------------------------
    # 6. True Possession Check
    # ------------------------------------------------------------------
    kept = True  # assume kept until proven otherwise
    player_team = row['team_name']

    # Iterate forward through events
    for i in range(event_idx + 1, len(match_events)):

        next_team = match_events.iloc[i]['team_name']

        # If opponent gains possession before player team does anything else → LOST
        if next_team != player_team:
            kept = False
            break

        # If player's team performs any action → KEPT
        if next_team == player_team:
            kept = True
            break

    kept_flags.append(kept)

# 7. Add to dataframe
under_pressure_passes['kept_possession'] = kept_flags

# 8. Filter kept balls
kept_under_pressure = under_pressure_passes[under_pressure_passes['kept_possession']]

# 9. Count
num_kept = kept_under_pressure.shape[0]
total = under_pressure_passes.shape[0]

print("Passes received under pressure:", total)
print("Passes kept under pressure:", num_kept)
print(f"Percentage kept: {num_kept/total*100:.2f}%")


# %%
matches = parser.match(1267, 107)   # AFCON 2023 competition
match_ids = matches["match_id"].tolist()
events = pd.concat([parser.event(m)[0] for m in match_ids], ignore_index=True)


# %%
cb_positions = ["Center Forward" ]
central_defenders = events[events["position_name"].isin(cb_positions)]



# %%
all_shots = central_defenders[central_defenders["type_name"] == "Shot"]
#all_shots["outcome_name"].value_counts()


# %%

players = all_shots["player_name"].unique()
on_target_outcomes =["Saved", "Goal", "Saved To Post"]
rows = []

for p in players:

    player_shots = all_shots[all_shots["player_name"] == p]
    total = len(player_shots)

    if total == 0:
        acc = 0
    else:
        on_target_count = len(
            player_shots[
                player_shots["outcome_name"].isin(on_target_outcomes)
            ]
        )
        acc = on_target_count / total * 100

    rows.append({
        "player": p,
        "total_shots": total,
        "on_target": on_target_count,
        "shot_accuracy_%": round(acc,2)
    })

shot_accuracy_df = pd.DataFrame(rows)

#shot_accuracy_df.head(53)


# %%


cb_positions = ["Center Forward" ]
central_defenders = events[events["position_name"].isin(cb_positions)]
cf_players = central_defenders["player_name"].unique()




# %%
def passes_under_pressure_box(player, events):

    # passes received
    rec = events[
        (events["type_name"]=="Pass") &
        (events["pass_recipient_name"]==player)
    ]

    # inside box
    rec = rec[
        (rec["end_x"].between(96,120)) &
        (rec["end_y"].between(18,62))
    ]

    ids = rec["id"].tolist()
    if len(ids)==0:
        return pd.Series(dtype=int), 0

    # sort
    sorted_ev = events.sort_values(["match_id","index"]).reset_index(drop=True)

    rows = sorted_ev[sorted_ev["id"].isin(ids)].index.tolist()

    actions = []

    for r in rows:
        match_id = sorted_ev.iloc[r]["match_id"]

        for nxt in range(r+1, len(sorted_ev)):
            ev = sorted_ev.iloc[nxt]

            if ev["match_id"] != match_id:
                break

            if ev["player_name"] == player:
                actions.append(ev)
                break

    df = pd.DataFrame(actions)

    under_pressure = df[df["under_pressure"]==True]

    per_match = under_pressure.groupby("match_id").size()
    avg = per_match.mean() if len(per_match)>0 else 0

    return per_match, avg


# %%
rows = []

for p in cf_players:

    per_match, avg = passes_under_pressure_box(p, events)

    rows.append({
        "player": p,
        "total": int(per_match.sum()),
        "num_matches": len(per_match),
        "avg_per_match": round(avg,2),
        "breakdown": per_match.to_dict()
    })

cf_results = pd.DataFrame(rows)
cf_results = cf_results.sort_values("avg_per_match", ascending=False)
pd.set_option("display.max_rows", None)

#cf_results.head(74)


# %%


cb_positions = ["Center Forward" ]
central_defenders = events[events["position_name"].isin(cb_positions)]
cf_players = central_defenders["player_name"].unique()




# %%
def kept_under_pressure(player, all_events):

    # ----------------------------
    # 1. Passes received
    # ----------------------------
    passes_to_player = all_events[
        (all_events["type_name"]=="Pass") &
        (all_events["pass_recipient_name"]==player)
    ]

    # under pressure only
    under_pressure = passes_to_player[
        passes_to_player["under_pressure"]==True
    ]

    if len(under_pressure)==0:
        return 0,0,0.0

    # sort
    under_pressure = under_pressure.sort_values(
        ["match_id","period","minute","second","index"]
    ).copy()

    kept_flags = []

    # ----------------------------------------------------------
    # iterate every pass reception
    # ----------------------------------------------------------
    for idx,row in under_pressure.iterrows():

        # all match events
        match_events = all_events[
            all_events["match_id"]==row["match_id"]
        ].sort_values(
            ["period","minute","second","index"]
        ).reset_index(drop=False)

        sb_index = row["index"]

        loc = match_events.index[
            match_events["index"]==sb_index
        ]

        if len(loc)==0:
            kept_flags.append(False)
            continue

        event_idx = loc[0]

        player_team = row["team_name"]

        kept = True

        # ------------------------------------------------------
        # Possession check
        # ------------------------------------------------------
        for i in range(event_idx+1, len(match_events)):

            next_team = match_events.iloc[i]["team_name"]

            # opponent wins ball
            if next_team != player_team:
                kept = False
                break

            # own team acts → kept
            if next_team == player_team:
                kept = True
                break

        kept_flags.append(kept)

    # ----------------------------------------------------------
    # results
    # ----------------------------------------------------------
    total = len(under_pressure)
    kept = sum(kept_flags)
    percent = (kept/total)*100 if total>0 else 0

    return total, kept, percent


# %%
rows = []

for p in cf_players:

    total, kept, percent = kept_under_pressure(p, all_events)

    rows.append({
        "player": p,
        "passes_under_pressure": total,
        "kept_under_pressure": kept,
        "percent_kept": round(percent,2)
    })

cf_pressure = pd.DataFrame(rows)

cf_pressure = cf_pressure.sort_values("percent_kept", ascending=False)

cf_pressure


# %%
df1 = shot_accuracy_df.merge(
    cf_results,
    on="player",
    how="outer"
)

# %%
df = df1.merge(
    cf_pressure,
    on="player",
    how="outer"
)

# %%
df.head(74  )

# %%
df = df.fillna(0)
df.isna().sum() 

# %%
metrics = [
    "shot_accuracy_%",
    "avg_per_match",
    "passes_under_pressure",
    "kept_under_pressure",
    "percent_kept"
]


# %%
import matplotlib.pyplot as plt
import numpy as np

players = df["player"]
y = np.arange(len(players))

bar_height = 0.15
n = len(metrics)

plt.figure(figsize=(10, max(6, len(players)*0.45)))

for i, m in enumerate(metrics):
    
    plt.barh(
        y + i*bar_height,
        df[m],
        height=bar_height,
        label=m.replace("_"," ")
    )

plt.yticks(y + bar_height*(n/2), players)
plt.xlabel("Value")
plt.title("Centre Forward Performance Comparison (AFCON 2023)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05,1))

plt.tight_layout()
plt.show()


# %%
import pandas as pd
import numpy as np

# copy numeric columns only
num_cols = df.columns.drop(["player","breakdown"])

# Z-score
df_z = df.copy()
df_z[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()


# %%
df_z["overall_score"] = df_z[num_cols].mean(axis=1)


# %%
df_z = df_z.sort_values("overall_score", ascending=False)
df_z
df_z.head(74)

# %%
nsue = "Emilio Nsue López"

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 30))

# Assign a unique color for each metric
metric_colors = {
    num_cols[i]: plt.cm.tab10(i)   # automatically picks different colors
    for i in range(len(num_cols))
}

for col in num_cols:

    color = metric_colors[col]     # get color for this metric

    plt.scatter(
        df_z[col],                 # Z-score values
        df_z["player"],            # y axis
        label=col.replace("_", " "),
        s=80,
        c=[color] * len(df_z),     # same color for all players of that metric
        alpha=0.8
    )

plt.title("Z-score Performance Comparison (Centre Forwards)", fontsize=18)
plt.xlabel("Z-score", fontsize=14)
plt.ylabel("Player", fontsize=14)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05,1))
plt.show()


# %%
plt.figure(figsize=(7,20))

colors = ["red" if p == nsue else "blue" for p in df_z["player"]]

plt.barh(df_z["player"], df_z["overall_score"], color=colors)

plt.title("Overall Centre Forward Performance (Z-score)")
plt.xlabel("Overall Z-score")
plt.grid(True)

plt.show()


# %%
df_z.head(74)

# %%
#%pip install streamlit pandas matplotlib seaborn numpy


# %%
df_z3 = df_z.drop(columns=["breakdown"])
df_z3.to_csv("data/df_z.csv", index=False)

# %%
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.markdown("""
The above data is used for the Streamlit dashboard below.
""")


# ----------------------------------------------------------
# Load your data
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/df_z.csv")     # your z-score or raw metrics
    return df

df = load_data()

# ----------------------------------------------------------
# Define metrics
# ----------------------------------------------------------
plot_cols = [
    "shot_accuracy_%",
    "avg_per_match",
    "passes_under_pressure",
    "kept_under_pressure",
    "percent_kept"
]

# Make sure all metrics numeric
df[plot_cols] = df[plot_cols].apply(pd.to_numeric, errors='coerce')

# ----------------------------------------------------------
# Radar chart function
# ----------------------------------------------------------
def radar_chart(data, title):
    
    labels = plot_cols
    num = len(labels)

    angles = np.linspace(0, 2*np.pi, num, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))

    for player, row in data.iterrows():
        vals = row.values
        vals = np.concatenate((vals, [vals[0]]))
        
        ax.plot(angles, vals, linewidth=2, label=player)
        ax.fill(angles, vals, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1))

    return fig

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.title("Centre Forward Performance Dashboard — AFCON 2023")

# ----------------------------------------------------------
# Player selection
# ----------------------------------------------------------
players = df["player"].unique().tolist()
selected = st.multiselect("Select players to compare", players, default=players[:2])

if len(selected) < 1:
    st.warning("Select at least one player.")
    st.stop()

# ----------------------------------------------------------
# Filter data
# ----------------------------------------------------------
comp = df[df["player"].isin(selected)].set_index("player")[plot_cols]

# ----------------------------------------------------------
# Display table
# ----------------------------------------------------------
st.subheader("Metrics Table")
st.dataframe(comp)

# ----------------------------------------------------------
# Radar chart
# ----------------------------------------------------------
st.subheader("Radar Comparison")
fig = radar_chart(comp, "Centre Forward Performance Profile")
st.pyplot(fig)

# ----------------------------------------------------------
# Additional summary
# ----------------------------------------------------------
st.subheader("Summary")
st.write("""
The radar chart compares key attacking and possession metrics:
- Shot accuracy (%) shows finishing efficiency.
- Average goals per match reflects scoring consistency.
- Passes under pressure & passes kept under pressure measure link-up quality.
- Percentage kept shows composure under defensive pressure.
""")



