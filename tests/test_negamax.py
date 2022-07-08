
def test_reorder_cols():
    from agents.agent_minimax.minimax import reorder_columns

    all_cols = [0, 1, 2, 3, 4, 5, 6]
    desired_order_all_cols = [3, 4, 2, 5, 1, 6, 0]
    five_cols = [0, 1, 2, 3, 4, 5]
    desired_order_five_cols = [3, 4, 2, 5, 1, 0]
    four_cols = [0, 1, 2, 3, 4]
    desired_order_four_cols = [2, 3, 1, 4, 0]
    three_cols = [2, 3, 4]
    desired_order_three_cols = [3, 4, 2]
    two_cols = [3, 5]
    desired_order_two_cols = [5, 3]
    one_col = [0]
    desired_order_one_cols = [0]

    assert desired_order_all_cols == reorder_columns(all_cols)
    assert desired_order_five_cols == reorder_columns(five_cols)
    assert desired_order_four_cols == reorder_columns(four_cols)
    assert desired_order_three_cols == reorder_columns(three_cols)
    assert desired_order_two_cols == reorder_columns(two_cols)
    assert desired_order_one_cols == reorder_columns(one_col)
