import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import chromadb
import uuid
from datetime import datetime

# ===================== 配置区 =====================
# 从Streamlit后台Secrets读取配置，无需修改代码
client_llm = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url=st.secrets["OPENAI_BASE_URL"]
)
MODEL_NAME = st.secrets.get("MODEL_NAME", "gpt-4o-mini")
EMBED_MODEL = st.secrets.get("EMBED_MODEL", "text-embedding-3-small")

# 向量库初始化
chroma_client = chromadb.PersistentClient(path="./chroma_db")
coll = chroma_client.get_or_create_collection(name="merger_collection")

CSV_FILE = "merger_projects.csv"

# 信息来源下拉选项
SOURCE_OPTIONS = ["客户一手", "中介机构", "其他"]
# 省份-市级列表（可自行扩充）
CITY_LIST = [
    "浙江省-杭州市", "浙江省-宁波市", "浙江省-温州市",
    "上海市-上海市", "江苏省-南京市", "江苏省-苏州市",
    "浙江省-绍兴市", "浙江省-嘉兴市", "浙江省-湖州市",
    "浙江省-金华市", "浙江省-台州市", "浙江省-衢州市",
    "浙江省-丽水市", "浙江省-舟山市"
]

# 腾讯文档标的库台账字段映射（对应你的并购撮合台账）
TARGET_LEDGER_COLS = [
    "序号", "项目公司全称", "公司属性", "是否上市公司", "交易诉求",
    "所属赛道", "主打产品", "区域", "所属分行", "项目联系人",
    "信息来源", "录入时间", "最新更新时间", "对接情况", "项目编号", "备注"
]
# 腾讯文档需求库台账字段映射
DEMAND_LEDGER_COLS = [
    "序号", "收购方全称", "交易诉求", "公司属性", "类别",
    "并购标的所属行业", "信息来源", "分行", "我行联系人",
    "录入时间", "录入人", "最近更新时间", "对接情况", "项目编号", "备注"
]

# AI系统提示词（精简版，节约token）
SYS_PROMPT = """
你是并购智能匹配助手，浙商银行内部使用。
声明：本链接仅供浙商银行内部使用，请勿外传。如需获取标的/并购方联系方式，请联系总行投行部资源中心虞田力、彭博。
任务：
1. 解析标的交易诉求文本，输出3部分内容：
①【标的推介文案】固定4段【项目名称】【投资亮点】【财务数据】【出售意向】，全文≤800字；上市公司强脱敏，非上市公司弱脱敏；信息缺失模块直接跳过，禁止编造数据。
②【所属赛道】：提炼1个核心赛道，例如功率半导体、储能、生物医药。
③【主打产品】：提炼核心产品，多个产品用「、」分隔；无则填写“未披露”。
输出严格JSON，key: introduction_copy, target_track, target_product

2. 解析并购需求文本，严格输出5行结构化内容：
【产业领域】
【地区偏好】
【财务要求】
【交易对价】
【其他要求】

3. 匹配打分：给定候选列表（最多20条），总分100分：赛道契合30分、交易规模25分、交易诉求20分、地域10分、基本面15分。按入参result_count返回TOP N（1~10）。
输出JSON，字段：match_result，数组，每个元素包含rank,total_score,project_no,project_status,introduction_copy。禁止输出任何额外文字、分项打分理由。
"""

# ===================== 初始化数据 =====================
def init_data():
    try:
        df = pd.read_csv(CSV_FILE)
    except:
        df = pd.DataFrame(columns=[
            "project_id", "project_name", "proj_type", "region",
            "source", "contact_person", "content_raw",
            "track", "main_product", "structured_text",
            "intro_copy", "status", "update_time", "create_time"
        ])
        df.to_csv(CSV_FILE, index=False)
    return df

def save_df(df):
    df.to_csv(CSV_FILE, index=False)

# 向量库：新增项目向量
def add_vector(pid:str, text:str, metadata:dict):
    try:
        emb = client_llm.embeddings.create(input=text, model=EMBED_MODEL).data[0].embedding
        coll.upsert(
            ids=[pid],
            embeddings=[emb],
            documents=[text],
            metadatas=[metadata]
        )
        return True
    except Exception as e:
        st.warning(f"向量写入失败：{e}")
        return False

# 向量库：删除项目向量
def del_vector(pid:str):
    try:
        coll.delete(ids=[pid])
    except:
        pass

# 向量召回，返回topN项目id
def vector_search(query_text:str, limit=20):
    try:
        emb_query = client_llm.embeddings.create(input=query_text, model=EMBED_MODEL).data[0].embedding
        res = coll.query(
            query_embeddings=[emb_query],
            n_results=limit
        )
        return res["ids"][0]
    except:
        return []

# LLM调用函数
def llm_call(user_prompt):
    resp = client_llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":"system", "content":SYS_PROMPT},
            {"role":"user", "content":user_prompt}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content

# 复制台账文本到剪贴板（一键入库，对齐腾讯文档列顺序）
def build_tencent_doc_row(row, ledger_type="标的"):
    """生成腾讯文档台账单行文本，用制表符分隔，粘贴到Excel/腾讯文档自动分列"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if ledger_type == "标的":
        line = "\t".join([
            str(row.name+1),
            str(row.get("project_name", "")),
            "",  # 公司属性
            "",  # 是否上市公司
            str(row.get("intro_copy", row.get("content_raw", ""))),
            str(row.get("track", "")),
            str(row.get("main_product", "")),
            str(row.get("region", "")),
            "",  # 所属分行
            str(row.get("contact_person", "")),
            str(row.get("source", "")),
            str(row.get("create_time", now)),
            str(row.get("update_time", now)),
            "",  # 对接情况
            str(row.get("project_id", "")),
            "",  # 备注
        ])
    else:
        line = "\t".join([
            str(row.name+1),
            str(row.get("project_name", "")),
            str(row.get("structured_text", row.get("content_raw", ""))),
            "",  # 公司属性
            "",  # 类别
            str(row.get("track", "")),
            str(row.get("source", "")),
            "",  # 分行
            str(row.get("contact_person", "")),
            str(row.get("create_time", now)),
            "",  # 录入人
            str(row.get("update_time", now)),
            "",  # 对接情况
            str(row.get("project_id", "")),
            "",  # 备注
        ])
    return line

# 从Excel/CSV导入存量台账数据
def import_ledger_from_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df_import = pd.read_csv(uploaded_file)
        else:
            df_import = pd.read_excel(uploaded_file)
        new_rows = []
        for _, row in df_import.iterrows():
            # 识别是标的还是需求表
            if "项目公司全称" in df_import.columns:
                ptype = "标的"
                pid_col = "项目编号"
                pid = str(row.get(pid_col, f"BGBD-{uuid.uuid4().hex[:8].upper()}"))
                pname = str(row.get("项目公司全称", ""))
                region = str(row.get("区域", ""))
                source = str(row.get("信息来源", "客户一手"))
                contact = str(row.get("项目联系人", ""))
                content = str(row.get("交易诉求", ""))
                track = str(row.get("所属赛道", ""))
                product = str(row.get("主打产品", ""))
                intro = content
                structured = ""
                status = str(row.get("状态", "项目有效"))
                create_time = str(row.get("录入时间", datetime.now().strftime("%Y-%m-%d %H:%M")))
                update_time = str(row.get("最新更新时间", create_time))
            else:
                ptype = "需求"
                pid_col = "项目编号"
                pid = str(row.get(pid_col, f"BGXQ-{uuid.uuid4().hex[:8].upper()}"))
                pname = str(row.get("收购方全称", ""))
                region = str(row.get("区域", ""))
                source = str(row.get("信息来源", "客户一手"))
                contact = str(row.get("我行联系人", ""))
                content = str(row.get("交易诉求", ""))
                track = str(row.get("并购标的所属行业", ""))
                product = ""
                intro = ""
                structured = content
                status = "项目有效"
                create_time = str(row.get("录入时间", datetime.now().strftime("%Y-%m-%d %H:%M")))
                update_time = str(row.get("最近更新时间", create_time))
            new_rows.append({
                "project_id": pid,
                "project_name": pname,
                "proj_type": ptype,
                "region": region,
                "source": source,
                "contact_person": contact,
                "content_raw": content,
                "track": track,
                "main_product": product,
                "structured_text": structured,
                "intro_copy": intro,
                "status": status,
                "update_time": update_time,
                "create_time": create_time
            })
            # 向量入库
            meta = {"type": ptype, "track": track, "status": status}
            add_vector(pid, content, meta)
        return pd.DataFrame(new_rows), len(new_rows)
    except Exception as e:
        st.error(f"导入失败：{e}")
        return None, 0

# ===================== 页面路由 =====================
st.set_page_config(page_title="并购智能匹配助手", layout="wide")
# 顶部全局提示条
st.warning("**内部资料·请勿外传**\n本链接仅供浙商银行内部使用，请勿外传。如需获取标的/并购方联系方式，请联系总行投行部资源中心虞田力、彭博。")

page = st.sidebar.selectbox("页面选择", ["并购标的录入", "并购需求录入", "管理员工作台"])
df = init_data()

# ===================== 页面1：并购标的录入 =====================
if page == "并购标的录入":
    st.header("并购标的录入（一键匹配并购方）")
    with st.form("form_target"):
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox("所属地区【必填】", CITY_LIST)
            source = st.selectbox("信息来源【必填】", SOURCE_OPTIONS)
            contact_person = st.text_input("项目联系人【必填】")
        with col2:
            proj_name = st.text_input("项目名称（选填）")
            reporter = st.text_input("填报人（选填）")
            contact_info = st.text_input("联系方式（选填）")
        content_raw = st.text_area("交易诉求文本", height=220, placeholder="""交易诉求要素提示（建议至少填写 4 项，缺失会显著影响匹配精度）
财务情况 · 如「2024 年营收 3.2 亿元、净利润 8900 万元」，请分年度填写
出售意向 · 出让比例（51%~100% / 整体出售）、是否接受业绩对赌、是否保留管理团队
所属赛道 · 如新能源、半导体、生物医药、高端装备、新材料
主打产品 · 如碱抛添加剂、三元正极材料、智能悬挂系统，多产品用「、」分隔
标的所处区域 · 统一写成「X省-X市」格式，如「浙江省-绍兴市」""")
        uploaded_file = st.file_uploader("上传文件（PDF/Word/Excel，选填）", type=["pdf","docx","xlsx"])
        submit_btn = st.form_submit_button("一键匹配并购方")

    if submit_btn:
        if not (region and source and contact_person and content_raw.strip()):
            st.error("所属地区、信息来源、项目联系人、交易诉求为必填项！")
        else:
            with st.spinner("正在AI提取赛道、主打产品并生成推介文案..."):
                prompt = f"【标的文本】\n{content_raw}"
                ai_out = llm_call(prompt)
                try:
                    res = json.loads(ai_out)
                    intro_copy = res["introduction_copy"]
                    track = res["target_track"]
                    main_product = res["target_product"]
                except Exception as e:
                    st.warning(f"AI JSON解析异常，使用原始文本：{e}")
                    intro_copy = ai_out
                    track = "待人工复核"
                    main_product = "待人工复核"

                # 生成项目编号
                pid = f"BGBD-{uuid.uuid4().hex[:8].upper()}"
                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                new_row = pd.DataFrame([{
                    "project_id": pid,
                    "project_name": proj_name,
                    "proj_type": "标的",
                    "region": region,
                    "source": source,
                    "contact_person": contact_person,
                    "content_raw": content_raw,
                    "track": track,
                    "main_product": main_product,
                    "structured_text": "",
                    "intro_copy": intro_copy,
                    "status": "项目有效",
                    "update_time": now,
                    "create_time": now
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                save_df(df)

                # 写入向量库
                meta = {"type":"标的", "track":track, "status":"项目有效"}
                add_vector(pid, content_raw, meta)
                st.success(f"项目已保存，项目编号：{pid}")
                st.subheader("AI自动提取结果")
                st.write(f"**所属赛道**：{track}")
                st.write(f"**主打产品**：{main_product}")
                st.text_area("脱敏推介文案", intro_copy, height=250)

            # ========== 向量召回 + LLM精打分 ==========
            st.divider()
            st.subheader("并购方匹配结果")
            top_n = st.slider("返回匹配结果数量", min_value=1, max_value=10, value=5)
            # 向量召回20条候选项目ID
            candidate_ids = vector_search(content_raw, limit=20)
            # 从df取出候选，过滤为【需求】类型，仅项目有效
            candidate_df = df[df["project_id"].isin(candidate_ids)]
            candidate_demand = candidate_df[(candidate_df["proj_type"]=="需求") & (candidate_df["status"]=="项目有效")].to_dict("records")
            st.info(f"向量召回候选数量：{len(candidate_demand)}")
            if len(candidate_demand) ==0:
                st.warning("暂无匹配的并购需求库数据")
            else:
                with st.spinner("AI正在精打分..."):
                    match_prompt = f"候选列表：{json.dumps(candidate_demand, ensure_ascii=False)}\n目标标的：{content_raw}\n返回TOP{top_n}"
                    match_json_str = llm_call(match_prompt)
                    try:
                        match_result = json.loads(match_json_str)
                        items = match_result["match_result"]
                        st.write("匹配结果按总分从高到低排序：")
                        for item in items:
                            st.markdown(f"**排名：{item['rank']}｜得分：{item['total_score']}｜项目编号：{item['project_no']}｜状态：{item.get('project_status','')}**")
                            st.text_area("推介文案", item["introduction_copy"], height=150, key=f"copy_{item['rank']}")
                    except Exception as e:
                        st.write("原始返回：", match_json_str)
                        st.exception(e)

# ===================== 页面2：并购需求录入 =====================
elif page == "并购需求录入":
    st.header("并购需求录入")
    with st.form("form_demand"):
        col1, col2 = st.columns(2)
        with col1:
            region_multi = st.multiselect("所属地区【必填，支持多选】", CITY_LIST)
            source = st.selectbox("信息来源【必填】", SOURCE_OPTIONS)
            contact_person = st.text_input("需求联系人【必填】")
        with col2:
            proj_name = st.text_input("需求名称（选填）")
            reporter = st.text_input("填报人（选填）")
            contact_info = st.text_input("联系方式（选填）")
        demand_raw = st.text_area("并购需求描述", height=220, placeholder="""并购需求要素提示（建议至少填写 4 项，缺失会显著影响匹配精度）
产业领域 · 如功率半导体、储能、生物医药、高端装备
地区偏好 · 可填写多区域，支持江浙沪、沿海等模糊地域描述
财务要求 · 如年净利润≥5000万元，营收≥2亿元，毛利率≥15%
交易对价 · 如估值区间2~8亿元，现金收购为主
其他要求 · 如希望保留原管理团队、接受业绩对赌、无重大法律瑕疵""")
        uploaded_file = st.file_uploader("上传文件（PDF/Word/Excel，选填）", type=["pdf","docx","xlsx"])
        submit_btn = st.form_submit_button("保存并结构化需求")

    if submit_btn:
        if not (region_multi and source and contact_person and demand_raw.strip()):
            st.error("所属地区、信息来源、需求联系人、需求描述为必填！")
        else:
            with st.spinner("AI正在结构化需求信息..."):
                ai_out = llm_call(f"【并购需求文本】\n{demand_raw}")
                structured_text = ai_out
                pid = f"BGXQ-{uuid.uuid4().hex[:8].upper()}"
                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                new_row = pd.DataFrame([{
                    "project_id": pid,
                    "project_name": proj_name,
                    "proj_type": "需求",
                    "region": ";".join(region_multi),
                    "source": source,
                    "contact_person": contact_person,
                    "content_raw": demand_raw,
                    "track": "",
                    "main_product": "",
                    "structured_text": structured_text,
                    "intro_copy": "",
                    "status": "项目有效",
                    "update_time": now,
                    "create_time": now
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                save_df(df)
                # 存入向量库
                meta = {"type":"需求", "track":"", "status":"项目有效"}
                add_vector(pid, demand_raw, meta)
                st.success(f"并购需求已保存，编号：{pid}")
                st.text_area("结构化并购需求", structured_text, height=220)

# ===================== 页面3：管理员工作台 =====================
elif page == "管理员工作台":
    st.header("并购项目管理工作台")
    # 统计卡片
    total = len(df)
    valid = len(df[df["status"]=="项目有效"])
    wait_review = len(df[df["status"]=="有效性待确认"])
    suspended = len(df[df["status"]=="项目搁置"])
    finished = len(df[df["status"]=="已成交"])
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1: st.metric("项目总数", total)
    with col2: st.metric("项目有效", valid)
    with col3: st.metric("有效性待确认", wait_review)
    with col4: st.metric("项目搁置", suspended)
    with col5: st.metric("已成交", finished)

    # ===== 导入存量台账功能 =====
    st.divider()
    st.subheader("📥 导入存量并购撮合台账")
    st.info("请将腾讯文档「并购撮合台账」导出为Excel(.xlsx)或CSV文件后上传，系统自动映射字段、入库并生成向量索引")
    uploaded_import = st.file_uploader("上传导出的台账文件", type=["xlsx","csv"])
    if st.button("开始导入"):
        if uploaded_import:
            new_df, count = import_ledger_from_file(uploaded_import)
            if new_df is not None and count >0:
                df = pd.concat([df, new_df], ignore_index=True).drop_duplicates(subset="project_id", keep="last")
                save_df(df)
                st.success(f"✅ 成功导入{count}条台账数据！")
                st.rerun()
        else:
            st.error("请先选择要导入的文件")

    # 筛选控件
    c1,c2,c3,c4 = st.columns(4)
    with c1: type_filter = st.selectbox("项目类型", ["全部","标的","需求"])
    with c2: status_filter = st.selectbox("项目状态", ["全部","项目有效","有效性待确认","项目搁置","已成交"])
    with c3: keyword = st.text_input("搜索关键词")
    with c4: show_wait_only = st.checkbox("仅看超期待确认")

    df_filter = df.copy()
    if type_filter != "全部":
        df_filter = df_filter[df_filter["proj_type"] == type_filter]
    if status_filter != "全部":
        df_filter = df_filter[df_filter["status"] == status_filter]
    if keyword:
        df_filter = df_filter[df_filter.apply(lambda x: keyword.lower() in str(x).lower(), axis=1)]
    if show_wait_only:
        df_filter = df_filter[df_filter["status"] == "有效性待确认"]

    st.divider()
    st.subheader("项目列表")
    show_cols = ["project_id","project_name","proj_type","track","main_product","region","source","contact_person","status","update_time"]
    st.dataframe(df_filter[show_cols], use_container_width=True)

    # 单行操作
    st.subheader("项目操作")
    if len(df_filter) ==0:
        st.warning("暂无数据")
    else:
        selected_pid = st.selectbox("选择项目编号", df_filter["project_id"].tolist())
        row_data = df[df["project_id"] == selected_pid].iloc[0]
        st.write(row_data)
        col_a,col_b,col_c,col_d,col_e = st.columns(5)
        with col_a:
            if st.button("打卡更新时间"):
                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                df.loc[df["project_id"] == selected_pid, "update_time"] = now
                save_df(df)
                st.success("更新时间已刷新")
        with col_b:
            if st.button("一键入库（生成台账行）"):
                ledger_type = "标的" if row_data["proj_type"]=="标的" else "需求"
                line = build_tencent_doc_row(row_data, ledger_type)
                st.success("✅ 已生成可粘贴至腾讯文档的台账行，全选下方文本复制后直接粘贴进表格即可：")
                st.code(line, language="text")
                st.download_button("下载台账行文件", line, file_name=f"台账行_{selected_pid}.txt")
        with col_c:
            if st.button("生成台账行"):
                ledger_type = "标的" if row_data["proj_type"]=="标的" else "需求"
                line = build_tencent_doc_row(row_data, ledger_type)
                st.code(line, language="text")
                st.download_button("下载台账文件", line, file_name=f"台账行_{selected_pid}.txt")
        with col_d:
            new_status = st.selectbox("修改状态", ["项目有效","有效性待确认","项目搁置","已成交"])
            if st.button("确认修改状态"):
                df.loc[df["project_id"] == selected_pid, "status"] = new_status
                save_df(df)
                st.success("状态已更新")
        with col_e:
            if st.button("删除项目"):
                df = df[df["project_id"] != selected_pid]
                del_vector(selected_pid)
                save_df(df)
                st.warning("项目已删除，向量库同步清除")
#（注：内容由AI生成）
