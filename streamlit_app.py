# app.py
import streamlit as st
import pandas as pd
import json
import re
import math
from pathlib import Path
from scipy.stats import pearsonr, spearmanr







# 1) Sayfa başlığı ve sekmeler
st.set_page_config(page_title="LATUC", layout="wide")
st.title("LATUC")

# ─── Top‐level description ───
st.markdown("""
Welcome to **interface of the LATUC**!  
This tool lets you search classical Ottoman poetry by text and POS‐tag,  
and visualize corpus‐wide statistics in the Analyzer.
Designed by Enes Yılandiloğlu.
Contact: enes.yilandiloglu (et) helsinki.fi
""")

main_tabs = st.tabs(["Explorer", "Analyzer"])
explorer_tab, analyzer_tab = main_tabs

# 2) Veri yükleme (cache)

# app.py ile aynı dizindeyse:
BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    # JSON ve CSV’ni repoya ekle: 
    #   - repo köküne koyduysan direkt:
    json_path = BASE_DIR / "LATUC.json"
    csv_path  = BASE_DIR / "LATUC_metadata.csv"
    

    # 1) JSON’u oku
    with open(json_path, "r", encoding="utf-8") as f:
        latuc = json.load(f)

    # 2) Metadata’yı oku
    metadata = pd.read_csv(csv_path, dtype=str)

    records = []
    # 3) Her file_name (fn) için döngü
    for fn, rec in latuc.items():
        matches = metadata.loc[metadata["ID"] == fn]
        if matches.empty:
            st.warning(f"Metadata not found for file '{fn}', skipping.")
            continue
        meta = matches.iloc[0].to_dict()

        # 4) Bu dosyaya ait her şiir için
        for poem_id, poem in rec.get("Poems", {}).items():
            poem_text = poem.get("Poem", "")
            lines     = poem_text.split("\n")
            tags      = poem.get("tags", [])

            # Eğer tags sayısı satır sayısından azsa boşlarla tamamla
            if len(tags) < len(lines):
                tags += [""] * (len(lines) - len(tags))

            # 5) Her satır için token_list ve upos_list oluştur
            token_list = [line.split() for line in lines]
            upos_list  = [
                tag_str.split() if tag_str else []
                for tag_str in tags
            ]

            # 6) Kayıt oluştur
            records.append({
                **meta,                 # ID, work_name, yüzyıl vs.
                "Title":      poem.get("Title"),
                "Meter":      poem.get("Meter"),
                "Poem":       poem_text,
                "lines":      lines,         # satır listesi
                "tags":       tags,          # satıra‐aligned UPOS dizeleri
                "token_list": token_list,    # satıra‐aligned token listeleri
                "upos_list":  upos_list      # satıra‐aligned UPOS listeleri
            })

    # 7) DataFrame olarak döndür
    return pd.DataFrame(records)

df = load_data()

# 3) Sidebar: filtreler ve arama
with st.sidebar:
    st.header("Filters")

    # Filtre multiselect'leri
    filter_fields = [
        ("work_name", "Work Name"),
        ("pen_name",  "Pen Name"),
        ("real_name", "Real Name"),
        ("viaf", "viaf"),
        ("century",   "Century"),
        ("gender",    "Gender"),
        ("rank",      "Rank")
    ]
    selections = {}
    for field, label in filter_fields:
        temp = df.copy()
        for other, _ in filter_fields:
            if other == field:
                continue
            sel = st.session_state.get(other, [])
            if sel:
                temp = temp[temp[other].isin(sel)]
        opts = sorted(temp[field].dropna().unique())
        selections[field] = st.multiselect(
            label, opts,
            default=st.session_state.get(field, []),
            key=field
        )

    # Filtre uygulanmış DataFrame
    filtered = df.copy()
    for field, sel in selections.items():
        if sel:
            filtered = filtered[filtered[field].isin(sel)]

    st.markdown("---")
    # Arama kutusu ve modu
    st.header("Text Search")
    query = st.text_input("Write your query:", key="query")
    search_mode = st.radio(
        "Query Mode:", ["Normal", "Regex"],
        index=0, key="search_mode"
    )

    st.markdown("---")
    st.header("POS Tag Search")
    # Eğer kullanıcı bir kelime girmişse, yalnızca o kelimenin tag'lerini listele
    if query:
        # satırları token/upos listesiyle explode et
        df_tags = df.explode(["token_list", "upos_list"])
        if search_mode == "Regex":
            # regex modunda kelime araması
            mask = df_tags["token_list"].str.contains(query, case=False, regex=True, na=False)
        else:
            # normal modda tam eşleşme
            mask = df_tags["token_list"].str.lower() == query.lower()
        # mask ile işaretli satırlardaki upos_list’leri düzleştirip tekil tag’leri alıyoruz
        relevant_tags = sorted({
            tag
            for upos_entry in df_tags.loc[mask, "upos_list"].dropna()
            for tag in upos_entry
        })

    else:
        # query yoksa tüm corpus tag'larını göster
        relevant_tags = sorted({
            tag
            for poem_upos in df["upos_list"]
            for line_upos in poem_upos
            for tag in line_upos
        })



    tag_query = st.selectbox("Choose a POS Tag (optional):", [""] + relevant_tags, key="tag_query")

# 4) Explorer Tab: içerik gösterimi# 4) Explorer Tab: içerik gösterimi
with explorer_tab:
    st.header("LATUC Explorer")

    st.markdown("""
    **Text & Tag Search**  
    Enter a word (or regex) and optionally select a POS‐tag  
    to filter lines in the corpus.
    """)

    # yalnızca metin veya tag sorgusu varsa arama yap
    use_regex = (search_mode == "Regex")
    df_exp = filtered.explode(["lines", "tags", "token_list", "upos_list"])

    if query or tag_query:

        # Eğer tag_query içinde boşluk varsa, 'word POS' formatı olarak al
        if tag_query and " " in tag_query:
            word_q, upos_q = tag_query.split(None, 1)
            # aynı pozisyonda hem token hem de UPOS eşleşmesi
            mask_pair = df_exp.apply(lambda r: any(
                t.lower() == word_q.lower() and u.lower() == upos_q.lower()
                for t, u in zip(r["token_list"], r["upos_list"])
            ), axis=1)

            mask_text = (
                df_exp["lines"]
                .str.contains(query, case=False, regex=use_regex, na=False)
                if query else True
            )
            results = df_exp[mask_pair & mask_text]

        else:
            # eskisi gibi ya metin ya da tekil POS araması
            mask_text = (
                df_exp["lines"]
                .str.contains(query, case=False, regex=use_regex, na=False)
                if query else True
            )
            mask_tag  = (
                df_exp["tags"]
                .str.contains(tag_query, case=False, regex=use_regex, na=False)
                if tag_query else True
            )
            results = df_exp[mask_text & mask_tag]

        if results.empty:
            st.info("No lines matching your search were found.")
        else:
            for work, wg in results.groupby("work_name"):
                st.subheader(work)
                for title, pg in wg.groupby("Title"):
                    for idx, (_, row) in enumerate(pg.iterrows()):
                        # metin vurgulama
                        if query:
                            pat = query if use_regex else re.escape(query)
                            line_hl = re.sub(f"({pat})", r"<mark>\1</mark>",
                                             row["lines"], flags=re.IGNORECASE)
                        else:
                            line_hl = row["lines"]

                        # tag vurgulama
                        if tag_query:
                            pat_t = tag_query if use_regex else re.escape(tag_query)
                            tags_hl = re.sub(f"({pat_t})", r"<mark>\1</mark>",
                                             row["tags"], flags=re.IGNORECASE)
                        else:
                            tags_hl = row["tags"]

                   
                        # --- 1) Satırı her zamanki gibi gösterin ---
                        st.markdown(line_hl, unsafe_allow_html=True)
                        
                        with st.expander(f"Show full poem (hit {idx+1})", expanded=False):
                            poem = row["Poem"]
                            # case‑insensitive regex for your search term
                            pattern = re.compile(re.escape(query), flags=re.IGNORECASE)

                            # wrap matches in a yellow span
                            highlighted = pattern.sub(
                                lambda m: f"<span style='background-color: yellow'>{m.group(0)}</span>",
                                poem
                            )
                            line_hl = re.sub(f"({pat})", r"<mark>\1</mark>",
                                             poem, flags=re.IGNORECASE)
                            # render with Markdown (convert newlines → line‑breaks)
                            st.markdown(line_hl.replace("\n", "  \n"), unsafe_allow_html=True)
                        

    else:
        # — Query boşsa —
        if filtered.empty:
            st.warning("No poems match the selected filters.")

        elif not any(selections.values()):
            # hiçbir filtre yoksa → tam korpus: alfabetik + sayfalı
            import math

            page_size = 20
            if "page" not in st.session_state:
                st.session_state.page = 1

            unique = (
                filtered[["century","real_name","work_name","Title","Poem"]]
                .drop_duplicates()
                .sort_values(
                    ["century","real_name","work_name","Title"],
                    ascending=[True,True,True,True]
                )
                .reset_index(drop=True)
            )

            total = unique.shape[0]
            total_pages = math.ceil(total / page_size)

            col1, _, col3 = st.columns([1,2,1])
            with col1:
                if st.button("← Previous Page") and st.session_state.page > 1:
                    st.session_state.page -= 1
            with col3:
                if st.button("Next Page →") and st.session_state.page < total_pages:
                    st.session_state.page += 1

            start = (st.session_state.page - 1) * page_size
            stop  = start + page_size
            page_df = unique.iloc[start:stop]

            last_cent, last_auth = None, None
            for _, row in page_df.iterrows():
                cent = row["century"]
                auth = row["real_name"]
                if cent != last_cent:
                    st.header(f"{cent}. Yüzyıl")
                    last_cent, last_auth = cent, None
                if auth != last_auth:
                    st.subheader(f"{auth} — {row['work_name']}")
                    last_auth = auth
                st.markdown(f"**{row['Title']}**")
                st.text(row["Poem"])
                st.divider()

            st.caption(f"Sayfa {st.session_state.page} / {total_pages}")

        else:
            # filtre uygulanmış, ancak query yoksa → sadece filtrelenmiş liste
            for work, wg in filtered.groupby("work_name"):
                st.subheader(work)
                for title, pg in wg.groupby("Title"):
                    st.markdown(f"**{title}**")
                    st.text(pg["Poem"].iloc[0])
                    st.divider()


# ——————————————————————————————
# LATUC Analyzer paneli
# ——————————————————————————————
with analyzer_tab:
    import altair as alt
    from scipy.stats import pearsonr

    st.header("LATUC Analyzer")

    st.markdown("""
    **Corpus Statistics**  
    View token frequencies, type‐token ratios,  
    and century‐by‐century trends across the entire dataset.
    """)

    # 1) Sidebar filtrelerine göre daraltılmış DataFrame
    df_an = filtered.copy()

    # 2) 📊 Temel Korpu­s İstatistikleri
    tokens = df_an["Poem"].str.split(r"\s+").explode().str.lower()
    stats = {
        "Total Tokens":    len(tokens),
        "Unique Types":    tokens.nunique(),
        "TTR":             round(tokens.nunique() / len(tokens), 4) if len(tokens) else 0,
        "Hapax Legomena":  (tokens.value_counts() == 1).sum(),
        "Line Count":      df_an["lines"].apply(len).sum(),
        "Poem Count":      len(df_an),
        "Meter Count":     df_an["Meter"].nunique(),
    }
    stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["Value"])
    st.subheader("Basic statistics for the corpus")
    st.dataframe(stats_df, use_container_width=True)

    # — Century Bazlı Work Sayısı —
    st.subheader("The number of works by century")
    work_counts = df_an.groupby("century")["work_name"].nunique().sort_index()
    st.bar_chart(work_counts)

    # — Century Bazlı Token & Type Sayıları ve TTR —
    st.subheader("The number of type and token by century")

    # explode ederek token listesi oluştur
    exploded = (
        df_an
        .assign(token=df_an["Poem"].str.lower().str.split(r"\s+"))
        .explode("token")
    )

    # yüzyıla göre token count ve type count
    token_counts = exploded.groupby("century")["token"].size()
    type_counts  = exploded.groupby("century")["token"].nunique()

    counts_df = pd.DataFrame({
        "Token_Count": token_counts,
        "Type_Count":  type_counts
    }).sort_index()

    # Altair ile yan yana (grouped) bar chart
    import altair as alt
    counts_melt = (
        counts_df
        .reset_index()
        .melt(id_vars="century", var_name="metric", value_name="count")
    )
    grouped_bar = (
        alt.Chart(counts_melt)
           .mark_bar()
           .encode(
               x=alt.X("century:N", title="Yüzyıl"),
               y=alt.Y("count:Q", title="Count"),
               color=alt.Color("metric:N", title="Metri̇k"),
               xOffset="metric:N"
           )
           .properties(width="container", height=300)
    )
    st.altair_chart(grouped_bar, use_container_width=True)

    # TTR için line plot
    # TTR için line plot
    st.subheader("TTR by century")
    ttr_by_century = (type_counts / token_counts).sort_index()
    st.line_chart(ttr_by_century)


    # 3) 📜 Vezin Frekansları (Top 10)
    st.subheader("The frequency of aruz meters (top 10)")
    vezin_counts = df_an["Meter"].value_counts()
    st.bar_chart(vezin_counts.head(10))
    if st.button("Show all meters"):
        full_vezin_df = (
            vezin_counts
            .reset_index()
            .rename(columns={"index": "Meter", "Meter": "Frequency"})
        )
        st.dataframe(full_vezin_df, use_container_width=True)

    # 4) 🚻 Yüzyıl × Cinsiyet Bazlı Yazar Dağılımı (table only)
    st.subheader("The distribution of the authors by century and gender")
    author_gender_cent = (
        df_an
        .groupby(["century", "gender"])["real_name"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    st.dataframe(author_gender_cent, use_container_width=True)


    # 5) 🏷️ Yüzyıl × Rank Bazlı Yazar Dağılımı (table only)
    st.subheader("The distribution of authors by century and rank")
    author_rank_cent = (
        df_an
        .groupby(["century", "rank"])["real_name"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    st.dataframe(author_rank_cent, use_container_width=True)


    # 6) 🔤 Kelime Arama ve Analizi
    st.subheader("Word based search and analysis")
    word_a = st.text_input("Write the first word (a):", key="analysis_word_a")
    word_b = st.text_input("Write the second word (b) (optional):", key="analysis_word_b")

    if word_a:
        # explode tokens
        df_tokens = (
            df_an
            .assign(token=df_an["Poem"].str.lower().str.split(r"\s+"))
            .explode("token")
        )
        a = word_a.lower()

        # — Tek Kelime Analizi —
        if not word_b:
            # 6.1) Frekans Tablosu (Kaç yazar, % yazar)
            author_count = (
                df_tokens[df_tokens["token"] == a]
                .groupby("century")["real_name"]
                .nunique()
            )
            total_authors_cent = df_an.groupby("century")["real_name"].nunique()
            pct_authors = (author_count / total_authors_cent * 100).fillna(0)
            auth_freq_df = pd.DataFrame({
                "Author_Count":       author_count,
                "Percent_Authors[%]": pct_authors
            }).sort_index()
            st.markdown("**Frequency table**")
            st.dataframe(auth_freq_df, use_container_width=True)


            # 6.2) Yazar Bazlı Kullanım (Real Name, Pen Name, Count, Relative)
            author_detail = (
                df_tokens[df_tokens["token"] == a]
                .groupby(["real_name", "pen_name"])
                .agg(Count=("token", "size"))
            )
            # her yazar için toplam token sayısı
            total_by_author = df_tokens.groupby(["real_name", "pen_name"]).size()
            author_detail["Relative_Freq"] = (author_detail["Count"] / total_by_author).fillna(0)

            author_detail_df = (
                author_detail
                .reset_index()
                .sort_values("Count", ascending=False)
                .rename(columns={"real_name": "Real_Name", "pen_name": "Pen_Name"})
            )

            st.markdown("**Author based use (Real name, pen name, absolute count, relative count)**")
            st.dataframe(author_detail_df[["Real_Name","Pen_Name","Count","Relative_Freq"]], use_container_width=True)


            # 6.3) Yazar Sayısı ve Yüzde (Cinsiyet × Yüzyıl)
            gender_count = (
                df
                .groupby(['century','gender'])
                .size()
                .unstack(fill_value=0)
                .sort_index()
            )

            # — 2) 'female' ve 'male' kolonlarını garantiye alın
            for g in ['female','male']:
                if g not in gender_count.columns:
                    gender_count[g] = 0

            # — 3) İsterseniz alfabetik sıraya sokun
            genders = sorted(gender_count.columns.tolist())

            # — 4) Yüzde hesaplaması
            total_per_century = gender_count.sum(axis=1)
            gender_pct = (
                gender_count
                .div(total_per_century, axis=0)
                .multiply(100)
                .fillna(0)
            )

            # — 5) Son tabloyu oluşturun
            cols = ['century'] + genders + [f"{g}_Pct" for g in genders]
            gender_table = (
                pd.concat([gender_count, gender_pct.add_suffix('_Pct')], axis=1)
                .reset_index()
                [cols]
            )


            st.markdown("**The number and percentage of authors**")
            st.dataframe(gender_table, use_container_width=True)


            # 6.4) Mention Sayısı (Cinsiyet × Yüzyıl)
            mention_gender = (
                df_tokens[df_tokens["token"] == a]
                .groupby(["century","gender"])
                .size()
                .unstack(fill_value=0)
                .sort_index()
            )
            st.markdown("**The number of hits across genders and time**")
            st.dataframe(mention_gender, use_container_width=True)


            # 6.5) Line Charts by Gender
            m_gender = mention_gender.reset_index().melt(
                id_vars="century", var_name="gender", value_name="mention_count"
            )
            st.altair_chart(
                alt.Chart(m_gender)
                   .mark_line(point=True)
                   .encode(
                       x="century:N", y="mention_count:Q", color="gender:N"
                   )
                   .properties(title="Absolute hits by gender"),
                use_container_width=True
            )
            norm_mention_gender = (
                mention_gender.div(
                    df_tokens.groupby(["century","gender"]).size()
                    .unstack(fill_value=0)
                    .sort_index()
                )
                .fillna(0)
            )
            n_gender = norm_mention_gender.reset_index().melt(
                id_vars="century", var_name="gender", value_name="norm_freq"
            )
            st.altair_chart(
                alt.Chart(n_gender)
                   .mark_line(point=True)
                   .encode(
                       x="century:N", y="norm_freq:Q", color="gender:N"
                   )
                   .properties(title="Normalized hits by gender"),
                use_container_width=True
            )

            # 6.6) Yazar Sayısı ve Yüzde (Rank × Yüzyıl)
            rank_count = (
                df_tokens[df_tokens["token"] == a]
                .groupby(["century","rank"])["real_name"]
                .nunique()
                .unstack(fill_value=0)
                .sort_index()
            )
            total_rank_auth_cent = (
                df_an
                .groupby(["century","rank"])["real_name"]
                .nunique()
                .unstack(fill_value=0)
                .sort_index()
            )
            rank_pct = (rank_count / total_rank_auth_cent * 100).fillna(0)
            rank_table = pd.concat(
                [rank_count, rank_pct.add_suffix('_Pct')],
                axis=1
            ).reset_index()
            st.markdown("**The number and percentage of authors across ranks and time**")
            st.dataframe(rank_table, use_container_width=True)


            # 6.7) Mention Sayısı (Rank × Yüzyıl)
            mention_rank = (
                df_tokens[df_tokens["token"] == a]
                .groupby(["century","rank"])
                .size()
                .unstack(fill_value=0)
                .sort_index()
            )
            st.markdown("**The number of hits across ranks and time**")
            st.dataframe(mention_rank, use_container_width=True)


            # 6.8) Line Charts by Rank
            m_rank = mention_rank.reset_index().melt(
                id_vars="century", var_name="rank", value_name="mention_count"
            )
            st.altair_chart(
                alt.Chart(m_rank)
                   .mark_line(point=True)
                   .encode(
                       x="century:N", y="mention_count:Q", color="rank:N"
                   )
                   .properties(title="Absolute hits by rank"),
                use_container_width=True
            )
            norm_mention_rank = (
                mention_rank.div(
                    df_tokens.groupby(["century","rank"]).size()
                    .unstack(fill_value=0)
                    .sort_index()
                )
                .fillna(0)
            )
            n_rank = norm_mention_rank.reset_index().melt(
                id_vars="century", var_name="rank", value_name="norm_freq"
            )
            st.altair_chart(
                alt.Chart(n_rank)
                   .mark_line(point=True)
                   .encode(
                       x="century:N", y="norm_freq:Q", color="rank:N"
                   )
                   .properties(title="Normalized hts by rank"),
                use_container_width=True
            )

            # 6.9) Korelasyon Değerleri
            mention_total = mention_gender.sum(axis=1)
            norm_total    = norm_mention_gender.sum(axis=1)
            r1, p1 = pearsonr(mention_total.index.astype(int), mention_total.values)
            r2, p2 = pearsonr(norm_total.index.astype(int), norm_total.values)
            st.markdown(f"**Raw Count Pearson r:** {r1:.4f}, p-value: {p1:.4g}")
            st.markdown(f"**Normalized Pearson r:** {r2:.4f}, p-value: {p2:.4g}")

        # — İki Kelime Analizi —
        else:
            b = word_b.lower()
            abs_a = df_tokens[df_tokens["token"] == a].groupby("century").size()
            abs_b = df_tokens[df_tokens["token"] == b].groupby("century").size()
            total_cent = df_tokens.groupby("century").size()
            rel_a = (abs_a / total_cent).fillna(0)
            rel_b = (abs_b / total_cent).fillna(0)

            # Century bazlı frekans tablosu
            ab_df = pd.DataFrame({
                f"{a}_abs": abs_a,
                f"{b}_abs": abs_b,
                f"{a}_rel": rel_a,
                f"{b}_rel": rel_b
            }).sort_index().fillna(0)
            st.markdown("**Frequency table for A/B (Century)**")
            st.dataframe(ab_df, use_container_width=True)


            # Proportion charts
            prop_raw  = (abs_a / (abs_a + abs_b) * 100).fillna(0)
            prop_norm = (rel_a / (rel_a + rel_b) * 100).fillna(0)
            prop_df = pd.DataFrame({
                "Raw_Prop_%":  prop_raw,
                "Norm_Prop_%": prop_norm
            }).sort_index()

            st.markdown("**Proportion A/(A+B) (Absolute)**")
            st.line_chart(prop_df["Raw_Prop_%"])
            st.markdown("**Proportion A/(A+B) (Normalized)**")
            st.line_chart(prop_df["Norm_Prop_%"])

            # — İki Kelime Analizi Gender & Rank Breakdown —
            # Cinsiyete Göre Proportion (A/(A+B))
            st.markdown("**Proportion (A/(A+B)) by gender and century**")
            abs_a_g = df_tokens[df_tokens["token"] == a].groupby(["century","gender"]).size()
            abs_b_g = df_tokens[df_tokens["token"] == b].groupby(["century","gender"]).size()
            sum_ab_g = abs_a_g.add(abs_b_g, fill_value=0)
            prop_gender = (
                (abs_a_g / sum_ab_g * 100)
                .unstack(fill_value=0)
                .sort_index()
            )
            st.dataframe(prop_gender, use_container_width=True)

            # Line chart by gender
            pg = prop_gender.reset_index().melt(
                id_vars="century", var_name="gender", value_name="prop_pct"
            )
            st.altair_chart(
                alt.Chart(pg)
                .mark_line(point=True)
                .encode(
                    x="century:N",
                    y="prop_pct:Q",
                    color="gender:N",
                    tooltip=["century","gender","prop_pct"]
                )
                .properties(title="Proportion A/(A+B) by gender")
                , use_container_width=True
            )

            # Rank’e Göre Proportion (A/(A+B))
            st.markdown("**Proportion (A/(A+B)) by rank and century**")
            abs_a_r = df_tokens[df_tokens["token"] == a].groupby(["century","rank"]).size()
            abs_b_r = df_tokens[df_tokens["token"] == b].groupby(["century","rank"]).size()
            sum_ab_r = abs_a_r.add(abs_b_r, fill_value=0)
            prop_rank = (
                (abs_a_r / sum_ab_r * 100)
                .unstack(fill_value=0)
                .sort_index()
            )
            st.dataframe(prop_rank, use_container_width=True)

            # Line chart by rank
            pr = prop_rank.reset_index().melt(
                id_vars="century", var_name="rank", value_name="prop_pct"
            )
            st.altair_chart(
                alt.Chart(pr)
                .mark_line(point=True)
                .encode(
                    x="century:N",
                    y="prop_pct:Q",
                    color="rank:N",
                    tooltip=["century","rank","prop_pct"]
                )
                .properties(title="Proportion A/(A+B) by rank")
                , use_container_width=True
            )

            # A/B Korelasyon
            if not word_b:
                # Tek kelime: raw count
                raw_series = abs_a.groupby("century").size()
                # İstersen normalize etmek için (% frequency):
                norm_series = (raw_series / total_cent * 100).fillna(0)

            else:
                # İki kelime: proportion = A / (A + B) * 100
                abs_a = abs_a.groupby("century").size()
                abs_b = abs_b.groupby("century").size()
                df_prop = pd.DataFrame({"A": abs_a, "B": abs_b}).fillna(0)
                # ham proportion
                raw_series = (df_prop["A"] / (df_prop["A"] + df_prop["B"]) * 100).fillna(0)
                # normalize edilmiş prop (örneğin yüzyıldaki tüm token sayısına göre de bölebilirsiniz)
                norm_series = (raw_series / total_cent * 100).fillna(0)

            def safe_corr(x, y):
                # En az 2 nokta ve değişkenlik varsa hesapla
                if len(x) >= 2 and y.nunique() > 1:
                    pr, pp = pearsonr(x, y)
                    sr, sp = spearmanr(x, y)
                else:
                    pr = pp = sr = sp = float("nan")
                return pr, pp, sr, sp
            idx = raw_series.index.astype(int)

            pr_raw, pp_raw, sr_raw, sp_raw = safe_corr(idx, raw_series)
            pr_norm, pp_norm, sr_norm, sp_norm = safe_corr(idx, norm_series)

            st.markdown("### Korelasyon Sonuçları")
            st.write(f"**Raw** — Pearson r={pr_raw:.3f} (p={pp_raw:.3f}), Spearman ρ={sr_raw:.3f} (p={sp_raw:.3f})")
            st.write(f"**Norm** — Pearson r={pr_norm:.3f} (p={pp_norm:.3f}), Spearman ρ={sr_norm:.3f} (p={sp_norm:.3f})")
