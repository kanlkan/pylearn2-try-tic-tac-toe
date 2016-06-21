#!/usr/bin/env ruby
#
# tic tac toe
#
# 0 : black (first)
# 1 ' white (second)
# 2 ' none
#

def show_board(array)
  p array[0][0].to_s + "," + array[0][1].to_s + "," + array[0][2].to_s
  p array[1][0].to_s + "," + array[1][1].to_s + "," + array[1][2].to_s
  p array[2][0].to_s + "," + array[2][1].to_s + "," + array[2][2].to_s
  p ""
end

def judge(array)
  ret = 2
  for stone in [0, 1] do
    for i in [0 ,1, 2] do
      if (array[i][0]==stone && array[i][1]==stone && array[i][2]==stone) ||
         (array[0][i]==stone && array[1][i]==stone && array[2][i]==stone)   then
        ret = stone
      end
    end
   
    if (array[0][0]==stone && array[1][1]==stone && array[2][2]==stone) ||
       (array[0][2]==stone && array[1][1]==stone && array[2][0]==stone)   then
      ret = stone
    end
  end
  return ret
end

loop_max = ARGV[0].to_i
#p "loop max=" + loop_max.to_s
cell_array = [] 
stone_array = [] 

loop_cnt = 0
until loop_cnt >= loop_max do
  cell_array = Array.new(9)
  stone_array = Array.new(3).map { Array.new(3, 2) }
  9.times do |num|
    cell_array[num] = num
  end

  i = 10
  history = []
  9.times do |num|
    #p i
    rnd = rand(i) - 1
    if num % 2 == 0
      stone_array[cell_array[rnd].divmod(3)[0]][cell_array[rnd].divmod(3)[1]] = 0
    else
      stone_array[cell_array[rnd].divmod(3)[0]][cell_array[rnd].divmod(3)[1]] = 1
    end
    history.push(cell_array[rnd])
    #show_board(stone_array)
    ret = judge(stone_array)
    if ret == 0 then
      history.push("win")    # black is winner.
      break
    elsif ret == 1 then
      history.push("lose")   # black is loser.
      break
    end
    
    cell_array.delete_at(rnd)

    i -= 1
  end
  p history.join(",")
  loop_cnt += 1
end


