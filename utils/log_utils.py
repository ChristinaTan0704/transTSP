def log_values(exe_cost, bl_val, loss, cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    if opts.problem == "ta":
        print('epoch: {}, train_batch_id: {}, avg_cost: {} avg_task_exe_cost {}'.format(epoch, batch_id, avg_cost, exe_cost.mean().item()))
    else:
        print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
    
        tb_logger.log_value('avg_cost', avg_cost, step)
        tb_logger.log_value('loss', loss, step)
        tb_logger.log_value('bl_val', bl_val.mean(), step)

        tb_logger.log_value('reinforce_loss/actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('log_likelihood', log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)

        if opts.problem == "ta":
            tb_logger.log_value('exe_cost', exe_cost.mean().item(), step)
