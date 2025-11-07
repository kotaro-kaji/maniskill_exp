#!/bin/bash

# === 設定 ===
SYNC_BRANCH="sync"

# === 引数チェック ===
if [ "$#" -ne 2 ]; then
  echo "❌ エラー: 引数が足りません。"
  echo "使用方法: ./git-squash-to.sh \"コミットメッセージ\" \"清書先のfeatureブランチ名\""
  echo "例: ./git-squash-to.sh \"Feat: 新機能\" \"feature/my-new-task\""
  exit 1
fi

# === 現在のブランチチェック ===
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$SYNC_BRANCH" ]; then
  echo "❌ エラー: このスクリプトは '$SYNC_BRANCH' ブランチから実行してください。"
  echo "現在は '$CURRENT_BRANCH' ブランチにいます。"
  exit 1
fi

COMMIT_MSG=$1
FEATURE_BRANCH=$2

echo "✅ 準備完了。以下の内容で清書を実行します:"
echo "   メッセージ: $COMMIT_MSG"
echo "   清書先ブランチ: $FEATURE_BRANCH"
echo "---"

# === ステップ1: Featureブランチに移動し、清書 ===

echo "==> $FEATURE_BRANCH ブランチに移動します..."
if ! git checkout $FEATURE_BRANCH; then
    echo "❌ エラー: $FEATURE_BRANCH ブランチに移動できませんでした。ブランチが存在するか確認してください。"
    exit 1
fi

echo "==> $FEATURE_BRANCH ブランチの最新版を取得します..."
if ! git pull origin $FEATURE_BRANCH; then
    echo "❌ エラー: $FEATURE_BRANCH の pull に失敗しました。コンフリクトを確認してください。"
    git checkout $SYNC_BRANCH # 失敗したらsyncに戻る
    exit 1
fi

echo "==> $SYNC_BRANCH ブランチの変更内容をマージします..."
if ! git merge --squash $SYNC_BRANCH; then
    echo "❌ エラー: 'merge --squash' に失敗しました。コンフリクトを手動で解決してください。"
    # この時点で手動介入が必要なため、スクリプトは停止する
    exit 1
fi

echo "==> きれいなコミットを作成します..."
git commit -m "$COMMIT_MSG"

echo "==> $FEATURE_BRANCH ブランチをプッシュします..."
if ! git push origin $FEATURE_BRANCH; then
    echo "❌ エラー: $FEATURE_BRANCH のプッシュに失敗しました。"
    exit 1
fi

echo "✅ $FEATURE_BRANCH への清書とプッシュが完了しました。"
echo "---"

# === ステップ2: 'sync' ブランチの後片付け ===

echo "==> $SYNC_BRANCH ブランチに戻ります..."
git checkout $SYNC_BRANCH

echo "==> ローカルの $SYNC_BRANCH を $FEATURE_BRANCH の最新状態にリセットします..."
git reset --hard $FEATURE_BRANCH

echo "==> リモートの $SYNC_BRANCH を強制プッシュでリセットします..."
if ! git push origin $SYNC_BRANCH --force; then
    echo "❌ エラー: $SYNC_BRANCH の強制プッシュに失敗しました。"
    exit 1
fi

echo "🎉 全ての処理が完了しました。"